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

from .problem import NoisyPoint
from .network import PointFlow


class TrainSample(NamedTuple):
    xt: Float[Array, "D"]
    dx: Float[Array, "D"]
    t: Float[Array, ""]
    y: Float[Array, "D"]


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
    problem: NoisyPoint
    flow: PointFlow
    tx: optax.GradientTransformation
    opt_state: optax.OptState
    flow_metrics: Welford
    key: Key[Array, ""]

    @classmethod
    def from_config(cls, args: argparse.Namespace) -> Self:
        """Build a fresh state from the problem and network in the run config."""
        key_sample, key_network, key_train = jr.split(jr.key(args.seed), 3)
        problem = NoisyPoint(**args.problem)

        # the network is shaped by one sample of the problem
        sample = problem.train_sample(key_sample)
        flow = PointFlow(
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
            key=key_train,
        )

    @eqx.filter_jit
    def train_step(self, batch: TrainSample) -> tuple[Self, Float[Array, ""]]:
        """Take one variance-normalized optimizer step on a batch, update running metrics."""
        flow_metrics = self.flow_metrics.update(batch.dx.astype(jnp.float32))

        # running variance, this batch included -- undefined on an empty Welford
        weight = 1.0 / jnp.maximum(flow_metrics.variance, 1e-12)

        def train_loss(flow: PointFlow) -> tuple[Float[Array, ""], Float[Array, ""]]:
            du_pred = jax.vmap(flow)(batch.xt, batch.t, batch.y)
            flow_loss = jnp.mean(jnp.square(du_pred - batch.dx))
            return weight * flow_loss, flow_loss

        (_, loss), grads = eqx.filter_value_and_grad(train_loss, has_aux=True)(
            self.flow
        )
        updates, opt_state = self.tx.update(
            grads, self.opt_state, eqx.filter(self.flow, eqx.is_inexact_array)
        )
        flow = eqx.apply_updates(self.flow, updates)

        state = self._replace(flow=flow, opt_state=opt_state, flow_metrics=flow_metrics)
        return state, loss

    @eqx.filter_jit
    def train_epoch(
        self, batch_size: int, n_steps: int
    ) -> tuple[Self, Float[Array, "S"]]:
        """Fuse n_steps (gen + train_step) pairs into one XLA dispatch via lax.scan."""
        dynamic, static = eqx.partition(self, eqx.is_array)

        def scan_step(dynamic: Self, _) -> tuple[Self, Float[Array, ""]]:
            state = eqx.combine(dynamic, static)
            key, key_batch = jr.split(state.key)
            batch = jax.vmap(state.problem.train_sample)(
                jr.split(key_batch, batch_size)
            )
            state, losses = state._replace(key=key).train_step(batch)
            return eqx.filter(state, eqx.is_array), losses

        dynamic, losses = jax.lax.scan(scan_step, dynamic, length=n_steps)
        return eqx.combine(dynamic, static), losses

    def save_to(
        self,
        checkpoints: ocp.CheckpointManager,
        epoch: int,
        loss_hist: Float[Array, "E S"],
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
    ) -> tuple[Self, int, Float[Array, "E S"] | None]:
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
    run_id = f"point-{args.config}"
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
        loss_history = np.full((epochs, args.log_interval), jnp.nan)

    pbar = tqdm(range(start_epoch, epochs), initial=start_epoch, total=epochs)
    for epoch in pbar:
        # one fused XLA dispatch for the whole epoch, instead of log_interval separate ones
        state, epoch_losses = state.train_epoch(args.batch_size, args.log_interval)
        loss_history[epoch] = jax.device_get(epoch_losses)

        # save a checkpoint and log the median of the epoch's losses
        state.save_to(checkpoints, epoch + 1, loss_history)
        flow_l = np.median(loss_history[epoch])
        pbar.set_postfix(flow=f"{flow_l:.5f}")
        print(f"[epoch {epoch + 1}/{epochs}] flow={flow_l:.5f}", flush=True)

        # redraw loss curve: median per epoch, shaded 10-90 percentile spread
        xs = np.arange(1, epoch + 2) * args.log_interval
        l_lo, l_med, l_up = np.percentile(
            loss_history[: epoch + 1], [10, 50, 90], axis=1
        )
        fig, ax = plt.subplots()
        ax.loglog(xs, l_med, label="flow", color="#2a86cf", lw=2)
        ax.fill_between(xs, l_lo, l_up, color="#2a86cf", alpha=0.15)
        ax.set(xlabel="step", ylabel="loss", title=f"training losses ({run_id})")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(frameon=False)
        fig.savefig(os.path.join(out_dir, "losses.pdf"), bbox_inches="tight")
        plt.close(fig)

    # orbax saves in a background thread: the last one has to land before we exit
    checkpoints.close()
    print(f"[done] {run_id} -> {out_dir}", flush=True)
