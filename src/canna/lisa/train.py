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
from .problem import LisaGB
from .network import LisaFlow


class TrainSample(NamedTuple):
    xt: Float[Array, "S 11"]
    dx: Float[Array, "S 11"]
    t: Float[Array, ""]
    y: Float[Array, "t f 3"]
    x_target: Float[Array, "S 11"]
    y_target: Float[Array, "t f 3"]
    f: Float[Array, ""]


def geodesic(
    problem: LisaGB,
    t: Float[Array, ""],
    x0: Float[Array, "S 11"],
    x1: Float[Array, "S 11"],
) -> Float[Array, "S 11"]:
    return problem.exp_map(x0, t * problem.log_map(x0, x1))


def path_position(t: Float[Array, ""], time_power: float) -> Float[Array, ""]:
    """How far along the geodesic the flow is at time t: s = 1 - (1 - t)^time_power.

    The fine structure of a posterior of width sigma lives at 1 - s < sigma / (prior
    width), the last ~1% of a uniform clock for the 0.2-bin f0 floor and the last ~3e-5
    for a loud source's true width, so a uniform clock barely trains it. Warping the clock
    with time_power > 1 keeps t uniform but spends more of it near the end of the path.
    The network is still conditioned on t, which stretches 1 - s by the same power, and
    the velocity it learns is the one along the path, d/ds, whose size does not shrink
    as s -> 1; the ODE multiplies it back by ds/dt (`path_speed`).
    """
    return t if time_power == 1 else 1 - (1 - t) ** time_power


def path_speed(t: Float[Array, ""], time_power: float) -> Float[Array, ""]:
    """ds/dt of `path_position`."""
    return jnp.ones_like(t) if time_power == 1 else time_power * (1 - t) ** (time_power - 1)


def train_sample(
    problem: LisaGB, key: Key[Array, ""], time_power: float = 1.0
) -> TrainSample:
    """Draw one training example: conditioning, a point on the geodesic, its velocity."""
    key_c, key_p, key_o, key_x0, key_t = jr.split(key, 5)
    f = problem.sample_f(key_c)
    p = problem.sample_physical(key_p, f)

    # noisy observation to condition on, clean one to reconstruct
    y = problem.preprocess(problem.sample_observation(key_o, p, f), f)
    y_target = problem.preprocess(problem.clean_signal(p, f), f)

    # sample and process flow quantities: t is uniform, the point sits at s(t) along the
    # path, and the target is the velocity along the path at s
    x0 = problem.sample_flow(key_x0, f)
    x1 = problem.physical_to_flow(p, f)
    t = jr.uniform(key_t, ())
    s = path_position(t, time_power)
    path = partial(geodesic, problem)
    xt = path(s, x0, x1)
    dx = jax.jacobian(path)(s, x0, x1)
    return TrainSample(xt=xt, dx=dx, t=t, y=y, x_target=x1, y_target=y_target, f=f)


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
    problem: LisaGB
    flow: LisaFlow
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
        problem = LisaGB(**args.problem)

        # the network is shaped by one sample of the problem
        sample = train_sample(problem, key_sample)
        flow = LisaFlow(
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
        self,
        batch: TrainSample,
        aux_weight: Float[Array, ""],
        lr_scale: Float[Array, ""] | float = 1.0,
    ) -> tuple[Self, Float[Array, "3"]]:
        """Take one variance-reweighted optimizer step on a batch, update running metrics.

        lr_scale multiplies the update. With no weight decay, Muon's and Adam's updates are
        linear in the lr, so this is exactly a scaled lr. Unlike an optax schedule, it keeps
        no step count in the optimizer state, so a resume with a fresh optimizer state still
        lands at the right point of the schedule.
        """
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

        def train_loss(flow: LisaFlow) -> tuple[Float[Array, ""], Float[Array, "3"]]:
            du_pred, u_pred, y_recon = jax.vmap(flow)(
                batch.xt, batch.t, batch.y, batch.f
            )
            flow_loss = jnp.mean(jnp.square(du_pred - batch.dx))
            x_loss = jnp.mean(jnp.square(self.problem.log_map(batch.x_target, u_pred)))
            y_loss = jnp.mean(jnp.square(y_recon - batch.y_target))

            losses = jnp.stack([flow_loss, x_loss, y_loss])
            return jnp.sum(weights * losses), losses

        (_, losses), grads = eqx.filter_value_and_grad(train_loss, has_aux=True)(
            self.flow
        )
        updates, opt_state = self.tx.update(
            grads, self.opt_state, eqx.filter(self.flow, eqx.is_inexact_array)
        )
        updates = jax.tree.map(lambda u: lr_scale * u, updates)
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
        self,
        aux_weight: Float[Array, ""],
        batch_size: int,
        n_steps: int,
        time_power: float = 1.0,
        lr_scale: Float[Array, ""] | float = 1.0,
    ) -> tuple[Self, Float[Array, "S 3"]]:
        """Fuse n_steps (gen + train_step) pairs into one XLA dispatch via lax.scan."""
        dynamic, static = eqx.partition(self, eqx.is_array)

        def scan_step(dynamic: Self, _) -> tuple[Self, Float[Array, "3"]]:
            state = eqx.combine(dynamic, static)
            key, key_batch = jr.split(state.key)
            batch = jax.vmap(partial(train_sample, state.problem, time_power=time_power))(
                jr.split(key_batch, batch_size)
            )
            state, losses = state._replace(key=key).train_step(batch, aux_weight, lr_scale)
            return eqx.filter(state, eqx.is_array), losses

        dynamic, losses = jax.lax.scan(scan_step, dynamic, length=n_steps)
        return eqx.combine(dynamic, static), losses

    def save_to(
        self,
        checkpoints: ocp.CheckpointManager,
        epoch: int,
        loss_hist: Float[Array, "E S 3"],
        save_opt_state: bool = True,
    ) -> None:
        """Save each array-carrying field, plus the losses, as its own named checkpoint item.

        The optimizer state is about two thirds of a checkpoint, and the manager holds the
        old checkpoint until the new one lands. save_opt_state=False leaves it out, so a
        quota that fits barely two full checkpoints still fits the run; a resume then
        restarts the optimizer's moments, which only matters at a job boundary.
        """
        skip = ("problem", "tx", "key") + (() if save_opt_state else ("opt_state",))
        checkpoints.save(
            epoch,
            args=ocp.args.Composite(
                loss_hist=ocp.args.ArraySave(loss_hist),
                # a bare key array is not a pytree StandardSave will take
                key=ocp.args.ArraySave(jr.key_data(self.key)),
                **{
                    name: ocp.args.StandardSave(eqx.filter(field, eqx.is_array))
                    for name, field in zip(self._fields, self)
                    if name not in skip
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

        # a checkpoint saved with save_opt_state=False has no optimizer state: keep the
        # fresh one this state was built with
        has_opt_state = (checkpoints.directory / str(latest_epoch) / "opt_state").exists()
        skip = ("problem", "tx", "key") + (() if has_opt_state else ("opt_state",))
        skeleton = {
            name: eqx.filter(field, eqx.is_array)
            for name, field in zip(self._fields, self)
            if name not in skip
        }
        # loss_hist and key are bare arrays, with no skeleton to take a sharding from, so
        # orbax would fall back to the one recorded at save time -- a device that need
        # not exist here, e.g. a GPU checkpoint read back on a CPU. Place them locally.
        local = ocp.ArrayRestoreArgs(
            sharding=jax.sharding.SingleDeviceSharding(jax.devices()[0])
        )
        restored = checkpoints.restore(
            latest_epoch,
            args=ocp.args.Composite(
                loss_hist=ocp.args.ArrayRestore(restore_args=local),
                key=ocp.args.ArrayRestore(restore_args=local),
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
        print(
            f"[checkpoint] resuming at epoch {latest_epoch}"
            + ("" if has_opt_state else " (no optimizer state saved: starting it afresh)"),
            flush=True,
        )
        return state, latest_epoch, restored["loss_hist"]


def run_dir(args: argparse.Namespace) -> Path:
    """Where a run keeps its checkpoints, plots and scorecards: outputs/lisa-<config>."""
    return args.output_dir / f"lisa-{args.config}"


def checkpoint_manager(out_dir: Path) -> ocp.CheckpointManager:
    """The run's checkpoints, keeping only the latest (orbax holds the old one while saving)."""
    return ocp.CheckpointManager(
        (out_dir / "checkpoints").absolute(),
        options=ocp.CheckpointManagerOptions(max_to_keep=1),
    )


def load_trained(args: argparse.Namespace) -> tuple[TrainState, int, Path]:
    """The latest trained state of the run args names, its epoch and its folder.

    For eval and scorecard. restore_from falls back to the fresh state when there is no
    checkpoint, which is right for a new training run but would quietly score an
    untrained network here, so stop instead.
    """
    out_dir = run_dir(args)
    checkpoints = checkpoint_manager(out_dir)
    state, epoch, _ = TrainState.from_config(args).restore_from(checkpoints)
    if epoch == 0:
        raise SystemExit(f"no checkpoint in {checkpoints.directory}: nothing to evaluate")
    return state, epoch, out_dir


def aux_weight_schedule(step: int, total_steps: int, warmup_frac: float) -> float:
    """Cosine anneal of the auxiliary heads, from 1 at the start to 0 after warmup_frac."""
    warmup_steps = warmup_frac * total_steps
    frac = min(step / warmup_steps, 1.0) if warmup_steps > 0 else 1.0
    return 0.5 + 0.5 * math.cos(math.pi * frac)


def lr_scale_schedule(epoch: int, epochs: int, cooldown_epochs: int) -> float:
    """1 until the last cooldown_epochs, then a linear decay to 0 (warmup-stable-decay).

    Each epoch takes the ramp's value at its midpoint, so a cooldown starts just below 1,
    ends just above 0 and averages 1/2.
    """
    start = epochs - cooldown_epochs
    if cooldown_epochs <= 0 or epoch < start:
        return 1.0
    return (epochs - epoch - 0.5) / cooldown_epochs


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
    parser.add_argument("--total_steps", type=int, default=500_000)
    parser.add_argument("--log_interval", type=int, default=1000)
    parser.add_argument("--warmup_frac", type=float, default=0.5)
    parser.add_argument(
        "--cooldown_steps",
        type=int,
        default=0,
        help="decay the lr linearly to 0 over the last N of total_steps (0: constant lr)",
    )
    parser.add_argument(
        "--time_power",
        type=float,
        default=1.0,
        help="flow clock warp, s = 1 - (1 - t)^p; 1 is the plain uniform clock",
    )
    parser.add_argument(
        "--ode_steps",
        type=int,
        default=None,
        help="eval and scorecard only: RK4 steps for sampling (default: eval.ODE_STEPS)",
    )
    parser.add_argument(
        "--n_random",
        type=int,
        default=None,
        help="scorecard only: random injections after the ten eval ones (default: scorecard.N_RANDOM)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=("bfloat16", "float32"),
        help="compute dtype for the network; params are kept in float32",
    )
    parser.add_argument("--muon", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--save_opt_state",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="include the optimizer state in checkpoints (about 2/3 of their size)",
    )
    parser.add_argument(
        "--require_checkpoint",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="stop instead of starting from scratch when there is no checkpoint to resume",
    )

    with open(config_args.config_root / f"{config_args.config}.yaml") as f:
        parser.set_defaults(**yaml.safe_load(f))
    return parser.parse_args()


if __name__ == "__main__":
    # headless: these run on cluster nodes with no display. Selecting the backend at
    # module level instead would hijack it for anything that merely imports the package
    # -- canna.lisa re-exports train_sample from here, so `from canna.lisa import LisaGB`
    # was silently switching notebooks to Agg and swallowing their figures.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    args = parse_args()

    # housekeeping
    out_dir = run_dir(args)
    run_id = out_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = checkpoint_manager(out_dir)
    print(f"JAX backend: {jax.default_backend()}", flush=True)
    print(f"devices: {jax.local_device_count()}", flush=True)
    print(f"run {run_id} -> {out_dir}", flush=True)

    # setup the training state
    state = TrainState.from_config(args)
    state, start_epoch, loss_history = state.restore_from(checkpoints)
    if args.require_checkpoint and start_epoch == 0:
        raise SystemExit(
            f"no checkpoint in {checkpoints.directory}: {args.config} only continues a run,"
            " see the header of its config"
        )
    epochs = args.total_steps // args.log_interval

    # the checkpoint carries the (epochs, log_interval) grid it was written on, and
    # total_steps or log_interval may well have changed since -- extending a run is how
    # you spend more wall time -- so re-allocate at the current shape and copy the
    # overlap in, rather than indexing off the end of the restored array
    restored, loss_history = loss_history, np.full(
        (epochs, args.log_interval, 3), np.nan
    )
    if restored is not None:
        rows = min(restored.shape[0], epochs)
        cols = min(restored.shape[1], args.log_interval)
        loss_history[:rows, :cols] = restored[:rows, :cols]

    cooldown_epochs = args.cooldown_steps // args.log_interval
    pbar = tqdm(range(start_epoch, epochs), initial=start_epoch, total=epochs)
    for epoch in pbar:
        aux_weight = aux_weight_schedule(epoch, epochs, args.warmup_frac)
        lr_scale = lr_scale_schedule(epoch, epochs, cooldown_epochs)

        # one fused XLA dispatch for the whole epoch, instead of log_interval separate
        # ones. aux_weight and lr_scale go in as arrays, not python floats: filter_jit treats
        # non-arrays as static, so a float re-traces the whole epoch every time it changes
        state, epoch_losses = state.train_epoch(
            jnp.asarray(aux_weight),
            args.batch_size,
            args.log_interval,
            args.time_power,
            jnp.asarray(lr_scale, dtype=jnp.float32),
        )
        loss_history[epoch] = jax.device_get(epoch_losses)

        # a non-finite loss has already poisoned the weights: stop, and leave the last
        # finite checkpoint in place instead of saving over it
        finite = np.isfinite(loss_history[epoch]).all(axis=-1)
        if not finite.all():
            checkpoints.close()
            raise SystemExit(
                f"[epoch {epoch + 1}] non-finite loss at step"
                f" {epoch * args.log_interval + int(np.argmin(finite))}, stopping"
            )

        # save a checkpoint and log the median of the epoch's losses
        state.save_to(checkpoints, epoch + 1, loss_history, args.save_opt_state)
        flow_l, x_l, y_l = np.median(loss_history[epoch], axis=0)
        pbar.set_postfix(flow=f"{flow_l:.5f}", x=f"{x_l:.5f}", y=f"{y_l:.5f}")
        print(
            f"[epoch {epoch + 1}/{epochs}] flow={flow_l:.5f} x={x_l:.5f}"
            f" y={y_l:.5f} aux_weight={aux_weight:.3f}"
            + (f" lr_scale={lr_scale:.4f}" if cooldown_epochs else ""),
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
