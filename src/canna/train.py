import math
from typing import NamedTuple, Self
from jaxtyping import Array, Float
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
from flax import nnx
from tqdm import tqdm
import matplotlib.pyplot as plt

from canna import networks, problems
from canna.problems import Problem, TrainSample


class TrainState(NamedTuple):
    problem: Problem
    flow: nnx.Module
    optimizer: nnx.Optimizer
    flow_metrics: nnx.metrics.Welford
    x_metrics: nnx.metrics.Welford
    y_metrics: nnx.metrics.Welford
    rngs: nnx.Rngs

    @classmethod
    def from_config(cls, args: argparse.Namespace) -> Self:
        """Build a fresh state from the problem and network in the run config."""
        rngs = nnx.Rngs(args.seed)

        # initialize the problem
        problem_class = getattr(problems, args.problem["class"])
        problem: Problem = problem_class(**args.problem.get("init_args", {}))

        # initialize the network, shaped by one sample of the problem
        network_class = getattr(networks, args.network["class"])
        sample = problem.train_sample(rngs())
        flow = network_class(
            **args.network.get("init_args", {}),
            x_shape=sample.xt.shape,
            y_shape=sample.y.shape,
            dtype=jnp.dtype(args.dtype),
            param_dtype=jnp.float32,
            rngs=rngs,
        )

        # initialize the optimizer
        optimizer = nnx.Optimizer(
            model=flow,
            wrt=nnx.Param,
            tx=optax.chain(
                optax.clip_by_global_norm(1.0),
                (optax.contrib.muon if args.muon else optax.adamw)(
                    args.learning_rate, weight_decay=args.weight_decay
                ),
            ),
        )

        return cls(
            problem=problem,
            flow=flow,
            optimizer=optimizer,
            flow_metrics=nnx.metrics.Welford(),
            x_metrics=nnx.metrics.Welford(),
            y_metrics=nnx.metrics.Welford(),
            rngs=rngs,
        )

    @nnx.jit(static_argnames=("batch_size",))
    def train_step(
        self, aux_weight: Float[Array, ""], batch_size: int
    ) -> Float[Array, "3"]:
        """Sample a batch, take one variance-reweighted optimizer step, update running metrics."""
        batch = jax.vmap(self.problem.train_sample)(jr.split(self.rngs(), batch_size))

        # variance as of the previous step's metrics; barely moves step to step
        target_var = jnp.array(
            [
                jnp.square(s.compute().standard_deviation)
                for s in (self.flow_metrics, self.x_metrics, self.y_metrics)
            ]
        )
        weights = jnp.array([1.0, aux_weight, aux_weight]) / jnp.maximum(
            target_var, 1e-12
        )

        def train_loss(flow: nnx.Module) -> tuple[Float[Array, ""], Float[Array, "3"]]:
            du_pred, u_pred, y_recon = flow(batch.xt, batch.y, batch.t)
            flow_loss = jnp.mean(jnp.square(du_pred - batch.dx))
            geometry = self.problem.geometry
            x_loss = jnp.mean(jnp.square(geometry.log_map(u_pred, batch.x_target)))
            y_loss = jnp.mean(jnp.square(y_recon - batch.y_target))

            losses = jnp.stack([flow_loss, x_loss, y_loss])
            return jnp.sum(weights * losses), losses

        (_, losses), grads = nnx.value_and_grad(train_loss, has_aux=True)(self.flow)
        self.optimizer.update(self.flow, grads)

        self.flow_metrics.update(values=batch.dx.astype(jnp.float32))
        self.x_metrics.update(values=batch.x_target.astype(jnp.float32))
        self.y_metrics.update(values=batch.y_target.astype(jnp.float32))

        return losses

    def save_to(
        self,
        checkpoints: ocp.CheckpointManager,
        epoch: int,
        loss_hist: Float[Array, "E S 3"],
    ) -> None:
        """Save each module, plus the losses, as its own named item in one checkpoint."""
        checkpoints.save(
            epoch,
            args=ocp.args.Composite(
                loss_hist=ocp.args.ArraySave(loss_hist),
                **{
                    name: ocp.args.StandardSave(nnx.state(module))
                    for name, module in zip(self._fields, self)
                    if name != "problem"
                },
            ),
        )

    def restore_from(
        self, checkpoints: ocp.CheckpointManager
    ) -> tuple[int, Float[Array, "E S 3"] | None]:
        """Update the modules in place, returning the epoch and losses to resume from."""
        latest_epoch = checkpoints.latest_step()
        if latest_epoch is None:
            return 0, None

        restored = checkpoints.restore(
            latest_epoch,
            args=ocp.args.Composite(
                loss_hist=ocp.args.ArrayRestore(),
                **{
                    name: ocp.args.StandardRestore(nnx.state(module))
                    for name, module in zip(self._fields, self)
                    if name != "problem"
                },
            ),
        )

        for name, module in zip(self._fields, self):
            if name == "problem":
                continue
            nnx.update(module, restored[name])

        print(f"[checkpoint] resuming at epoch {latest_epoch}", flush=True)
        return latest_epoch, restored["loss_hist"]


@nnx.jit(static_argnames=("batch_size",))
def sample_batch(problem: Problem, rngs: nnx.Rngs, batch_size: int) -> TrainSample:
    return jax.vmap(problem.train_sample)(jr.split(rngs(), batch_size))


def aux_weight_schedule(step: int, total_steps: int, warmup_frac: float) -> float:
    """Cosine anneal of the auxiliary heads, from 1 at the start to 0 after warmup_frac."""
    warmup_steps = warmup_frac * total_steps
    frac = min(step / warmup_steps, 1.0) if warmup_steps > 0 else 1.0
    return 0.5 + 0.5 * math.cos(math.pi * frac)


if __name__ == "__main__":
    # --config_root has to resolve before the run yaml can be read for defaults
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", required=True, help="name of a run .yaml")
    config_parser.add_argument(
        "--config_root", type=Path, default=Path(__file__).parent / "configs"
    )
    config_args, _ = config_parser.parse_known_args()

    # the run config is a .yaml that sets defaults for the CLI, and can be overridden by it
    parser = argparse.ArgumentParser(parents=[config_parser])
    parser.add_argument("--output_dir", type=Path, default=Path("outputs"))
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--total_steps", type=int, default=500_000)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--warmup_frac", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=("bfloat16", "float16", "float32"),
        help="compute dtype for the network; params are kept in float32",
    )
    parser.add_argument("--muon", action=argparse.BooleanOptionalAction, default=True)

    with open(config_args.config_root / f"{config_args.config}.yaml") as f:
        parser.set_defaults(**yaml.safe_load(f))

    args = parser.parse_args()

    # housekeeping
    run_id = args.config
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
    start_epoch, loss_history = state.restore_from(checkpoints)
    epochs = args.total_steps // args.log_interval
    if loss_history is None:
        loss_history = np.full((epochs, args.log_interval, 3), jnp.nan)

    pbar = tqdm(range(start_epoch, epochs), initial=start_epoch, total=epochs)
    for epoch in pbar:
        aux_weight = aux_weight_schedule(epoch, epochs, args.warmup_frac)

        # the losses stay on device for the whole epoch, then come back in one transfer
        epoch_losses = [
            state.train_step(aux_weight, args.batch_size)
            for _ in range(args.log_interval)
        ]
        loss_history[epoch] = jax.device_get(jnp.stack(epoch_losses))

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
