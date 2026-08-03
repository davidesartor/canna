from functools import partial
import math
from typing import NamedTuple, Self
from jaxtyping import Array, Float, Key
from pathlib import Path
import os
import argparse
import yaml

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
import orbax.checkpoint as ocp
import equinox as eqx
from tqdm import tqdm
import matplotlib.pyplot as plt

from .problem import NoisySinusoid
from .network import SinusoidFlow


class TrainSample(NamedTuple):
    xt: Float[Array, "S 4"]
    dx: Float[Array, "S 4"]
    t: Float[Array, ""]
    y: Float[Array, "t f 2"]
    x_target: Float[Array, "S 4"]
    y_target: Float[Array, "t f 2"]


def train_sample(problem: NoisySinusoid, key: Key[Array, ""]) -> TrainSample:
    """Draw one training example: conditioning, a point on the geodesic, its velocity."""
    key_p, key_o, key_x0, key_t = jr.split(key, 4)
    p = problem.sample_physical(key_p)

    # noisy observation to condition on, clean one to reconstruct
    y = problem.preprocess(problem.sample_observation(key_o, p))
    y_target = problem.preprocess(problem.clean_signal(p))

    # sample and process flow quantities
    x0 = problem.sample_point(key_x0)
    x1 = problem.physical_to_flow(p)
    t = jr.uniform(key_t, ())
    xt = problem.geodesic(t, x0, x1)
    dx = jax.jacobian(problem.geodesic)(t, x0, x1)
    return TrainSample(xt=xt, dx=dx, t=t, y=y, x_target=x1, y_target=y_target)


class Welford(NamedTuple):
    """Running count/mean/sum-of-squares over every value ever passed to update."""

    count: Float[Array, ""]
    mean: Float[Array, ""]
    m2: Float[Array, ""]

    @classmethod
    def empty(cls) -> Self:
        zero = jnp.zeros((), jnp.float32)
        return cls(count=zero, mean=zero, m2=zero)

    def update(self, values: Float[Array, "..."]) -> Self:
        batch_count = jnp.asarray(values.size, jnp.float32)
        batch_mean = jnp.mean(values)
        batch_m2 = jnp.sum(jnp.square(values - batch_mean))

        delta = batch_mean - self.mean
        count = self.count + batch_count
        return Welford(
            count=count,
            mean=self.mean + delta * batch_count / count,
            m2=self.m2
            + batch_m2
            + jnp.square(delta) * self.count * batch_count / count,
        )

    @property
    def variance(self) -> Float[Array, ""]:
        return self.m2 / self.count


class TrainState(NamedTuple):
    problem: NoisySinusoid
    flow: SinusoidFlow
    tx: optax.GradientTransformation
    opt_state: optax.OptState
    flow_metrics: Welford
    x_metrics: Welford
    y_metrics: Welford
    key: Key[Array, ""]

    @classmethod
    def from_config(cls, args: argparse.Namespace) -> Self:
        """Build a fresh state from the problem and network in the run config."""
        key_sample, key_network, key_train = jr.split(jr.key(args.seed), 3)
        problem = NoisySinusoid(**args.problem)

        # the network is shaped by one sample of the problem
        sample = train_sample(problem, key_sample)
        flow = SinusoidFlow(
            **args.network,
            x_shape=sample.xt.shape,
            y_shape=sample.y.shape,
            dtype=jnp.dtype(args.dtype),
            param_dtype=jnp.float32,
            key=key_network,
        )

        tx = optax.chain(
            optax.clip_by_global_norm(1.0),
            (optax.contrib.muon if args.muon else optax.adamw)(
                args.learning_rate, weight_decay=args.weight_decay
            ),
        )

        return cls(
            problem=problem,
            flow=flow,
            tx=tx,
            opt_state=tx.init(eqx.filter(flow, eqx.is_inexact_array)),
            flow_metrics=Welford.empty(),
            x_metrics=Welford.empty(),
            y_metrics=Welford.empty(),
            key=key_train,
        )

    @eqx.filter_jit
    def train_step(
        self, batch: TrainSample, aux_weight: Float[Array, ""]
    ) -> tuple[Self, Float[Array, "3"]]:
        """Take one variance-reweighted optimizer step on a batch, update running metrics."""
        flow_metrics = self.flow_metrics.update(batch.dx.astype(jnp.float32))
        x_metrics = self.x_metrics.update(batch.x_target.astype(jnp.float32))
        y_metrics = self.y_metrics.update(batch.y_target.astype(jnp.float32))

        # running variance, this batch included -- undefined on an empty Welford
        target_var = jnp.array(
            [m.variance for m in (flow_metrics, x_metrics, y_metrics)]
        )
        weights = jnp.array([1.0, aux_weight, aux_weight]) / jnp.maximum(
            target_var, 1e-12
        )

        def train_loss(
            flow: SinusoidFlow,
        ) -> tuple[Float[Array, ""], Float[Array, "3"]]:
            du_pred, u_pred, y_recon = jax.vmap(flow)(batch.xt, batch.t, batch.y)
            flow_loss = jnp.mean(jnp.square(du_pred - batch.dx))
            x_loss = jnp.mean(jnp.square(self.problem.log_map(u_pred, batch.x_target)))
            y_loss = jnp.mean(jnp.square(y_recon - batch.y_target))

            losses = jnp.stack([flow_loss, x_loss, y_loss])
            return jnp.sum(weights * losses), losses

        (_, losses), grads = eqx.filter_value_and_grad(train_loss, has_aux=True)(
            self.flow
        )
        updates, opt_state = self.tx.update(
            grads, self.opt_state, eqx.filter(self.flow, eqx.is_inexact_array)
        )
        flow = eqx.apply_updates(self.flow, updates)

        state = self._replace(
            flow=flow,
            opt_state=opt_state,
            flow_metrics=flow_metrics,
            x_metrics=x_metrics,
            y_metrics=y_metrics,
        )
        return state, losses

    @eqx.filter_jit
    def train_epoch(
        self, aux_weight: Float[Array, ""], batch_size: int, n_steps: int
    ) -> tuple[Self, Float[Array, "S 3"]]:
        """Fuse n_steps (gen + train_step) pairs into one XLA dispatch via lax.scan."""
        dynamic, static = eqx.partition(self, eqx.is_array)

        def scan_step(dynamic: Self, _) -> tuple[Self, Float[Array, "3"]]:
            state = eqx.combine(dynamic, static)
            key, key_batch = jr.split(state.key)
            batch = jax.vmap(partial(train_sample, state.problem))(
                jr.split(key_batch, batch_size)
            )
            state, losses = state._replace(key=key).train_step(batch, aux_weight)
            return eqx.filter(state, eqx.is_array), losses

        dynamic, losses = jax.lax.scan(scan_step, dynamic, length=n_steps)
        return eqx.combine(dynamic, static), losses

    def save_to(
        self,
        checkpoints: ocp.CheckpointManager,
        epoch: int,
        loss_hist: Float[Array, "E S 3"],
    ) -> None:
        """Save each array-carrying field, plus the losses, as its own named checkpoint item."""
        checkpoints.save(
            epoch,
            args=ocp.args.Composite(
                loss_hist=ocp.args.ArraySave(loss_hist),
                # a bare key array is not a pytree StandardSave will take
                key=ocp.args.ArraySave(jr.key_data(self.key)),
                **{
                    name: ocp.args.StandardSave(eqx.filter(field, eqx.is_array))
                    for name, field in zip(self._fields, self)
                    if name not in ("problem", "tx", "key")
                },
            ),
        )

    def restore_from(
        self, checkpoints: ocp.CheckpointManager
    ) -> tuple[Self, int, Float[Array, "E S 3"] | None]:
        """Return the state, epoch and losses to resume from, or self at epoch 0."""
        latest_epoch = checkpoints.latest_step()
        if latest_epoch is None:
            return self, 0, None

        skeleton = {
            name: eqx.filter(field, eqx.is_array)
            for name, field in zip(self._fields, self)
            if name not in ("problem", "tx", "key")
        }
        restored = checkpoints.restore(
            latest_epoch,
            args=ocp.args.Composite(
                loss_hist=ocp.args.ArrayRestore(),
                key=ocp.args.ArrayRestore(),
                **{
                    name: ocp.args.StandardRestore(tree)
                    for name, tree in skeleton.items()
                },
            ),
        )

        state = self._replace(
            key=jr.wrap_key_data(restored["key"]),
            **{
                name: eqx.combine(restored[name], getattr(self, name))
                for name in skeleton
            },
        )
        print(f"[checkpoint] resuming at epoch {latest_epoch}", flush=True)
        return state, latest_epoch, restored["loss_hist"]


def aux_weight_schedule(step: int, total_steps: int, warmup_frac: float) -> float:
    """Cosine anneal of the auxiliary heads, from 1 at the start to 0 after warmup_frac."""
    warmup_steps = warmup_frac * total_steps
    frac = min(step / warmup_steps, 1.0) if warmup_steps > 0 else 1.0
    return 0.5 + 0.5 * math.cos(math.pi * frac)


def parse_args() -> argparse.Namespace:
    """Read the named run .yaml as argparse defaults, so any CLI flag overrides it."""
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default="B", help="name of a run .yaml")
    config_parser.add_argument(
        "--config_root", type=Path, default=Path(__file__).parent / "configs"
    )
    config_args, _ = config_parser.parse_known_args()

    parser = argparse.ArgumentParser(parents=[config_parser])
    parser.add_argument("--output_dir", type=Path, default=Path("outputs"))
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--total_steps", type=int, default=10_000)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--warmup_frac", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=("bfloat16", "float32"),
        help="compute dtype for the network; params are kept in float32",
    )
    parser.add_argument("--muon", action=argparse.BooleanOptionalAction, default=True)

    with open(config_args.config_root / f"{config_args.config}.yaml") as f:
        parser.set_defaults(**yaml.safe_load(f))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # housekeeping
    run_id = f"sinusoid-{args.config}"
    out_dir: Path = args.output_dir / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = ocp.CheckpointManager(
        (out_dir / "checkpoints").absolute(),
        options=ocp.CheckpointManagerOptions(max_to_keep=1),
    )
    print(f"JAX backend: {jax.default_backend()}", flush=True)
    print(f"devices: {jax.local_device_count()}", flush=True)
    print(f"run {run_id} -> {out_dir}", flush=True)

    # setup the training state
    state = TrainState.from_config(args)
    state, start_epoch, loss_history = state.restore_from(checkpoints)
    epochs = args.total_steps // args.log_interval
    if loss_history is None:
        loss_history = np.full((epochs, args.log_interval, 3), jnp.nan)

    pbar = tqdm(range(start_epoch, epochs), initial=start_epoch, total=epochs)
    for epoch in pbar:
        aux_weight = aux_weight_schedule(epoch, epochs, args.warmup_frac)

        # one fused XLA dispatch for the whole epoch, instead of log_interval separate ones
        state, epoch_losses = state.train_epoch(
            aux_weight, args.batch_size, args.log_interval
        )
        loss_history[epoch] = jax.device_get(epoch_losses)

        # save a checkpoint and log the median of the epoch's losses
        state.save_to(checkpoints, epoch + 1, loss_history)
        flow_l, x_l, y_l = np.median(loss_history[epoch], axis=0)
        pbar.set_postfix(flow=f"{flow_l:.5f}", x=f"{x_l:.5f}", y=f"{y_l:.5f}")
        print(
            f"[epoch {epoch + 1}/{epochs}] flow={flow_l:.5f} x={x_l:.5f}"
            f" y={y_l:.5f} aux_weight={aux_weight:.3f}",
            flush=True,
        )

        # redraw loss curve: median per epoch, shaded 10-90 percentile spread
        xs = np.arange(1, epoch + 2) * args.log_interval
        l_lo, l_med, l_up = np.percentile(
            loss_history[: epoch + 1], [10, 50, 90], axis=1
        )
        fig, ax = plt.subplots()
        # keys are in the column order of the stacked losses
        loss_colors = {"flow": "#2a86cf", "x": "#1a9e6a", "y": "#8a4bd0"}
        for i, (name, color) in enumerate(loss_colors.items()):
            ax.loglog(xs, l_med[:, i], label=name, color=color, lw=2)
            ax.fill_between(xs, l_lo[:, i], l_up[:, i], color=color, alpha=0.15)
        ax.set(xlabel="step", ylabel="loss", title=f"training losses ({run_id})")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(frameon=False)
        fig.savefig(os.path.join(out_dir, "losses.pdf"), bbox_inches="tight")
        plt.close(fig)

    # orbax saves in a background thread: the last one has to land before we exit
    checkpoints.close()
    print(f"[done] {run_id} -> {out_dir}", flush=True)
