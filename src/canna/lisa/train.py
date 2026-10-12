from functools import partial
import time
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
    y: Float[Array, "F 6"]
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

    # the noisy observation to condition on
    y = problem.preprocess(problem.sample_observation(key_o, p, f), f)

    # sample and process flow quantities: t is uniform, the point sits at s(t) along the
    # path, and the target is the velocity along the path at s
    x0 = problem.sample_flow(key_x0, f)
    x1 = problem.physical_to_flow(p, f)
    t = jr.uniform(key_t, ())
    s = path_position(t, time_power)
    path = partial(geodesic, problem)
    xt = path(s, x0, x1)
    dx = jax.jacobian(path)(s, x0, x1)
    return TrainSample(xt=xt, dx=dx, t=t, y=y, f=f)


# Muon takes the matrices of these fields, AdamW everything else: the encodings (the PLE
# maps and the token positions), the last layer of the velocity head, and anything 1-D
MUON_FIELDS = ("backbone", "t_embed", "t_mlp", "f_mlp", "x_modulation")


def muon_dimensions(params: LisaFlow) -> LisaFlow:
    """Muon's view of each parameter: the matrix axes of a weight it takes, None otherwise.

    optax's default rule takes exactly the 2-D arrays. That missed every weight of the
    MMDiT blocks, which are stacked into (blocks, out, in) and (blocks, 3, out, in) arrays,
    and sent them to Adam, while it orthogonalised the stacked (blocks, D) biases across
    blocks -- the case in every run up to R1 and B2. Here the last two axes of an
    eqx.nn.Linear weight are the matrix, (out, in), and any axes before them are batch
    axes, each slice orthogonalised on its own.
    """

    def dims(path, leaf):
        names = [k.name for k in path if isinstance(k, jax.tree_util.GetAttrKey)]
        hidden = names[0] in MUON_FIELDS or names[:2] == ["x_unembed", "linear1"]
        if names[-1] != "weight" or not hidden:
            return None
        return optax.contrib.MuonDimensionNumbers(
            reduction_axis=leaf.ndim - 1, output_axis=leaf.ndim - 2
        )

    return jax.tree_util.tree_map_with_path(dims, params)


def optimizer(args: argparse.Namespace) -> optax.GradientTransformation:
    """Schedule-free Muon, with AdamW for the parameters Muon does not take.

    Schedule-free (Defazio et al. 2024, "The Road Less Scheduled", arXiv:2405.15682) keeps
    a fast sequence z, takes the gradient at y = (1 - b1) z + b1 x, and evaluates the
    average x, weighted by the squared lr. It needs no cooldown and no total length, only a
    warmup. Muon keeps its own momentum inside it, as SF-Muon does in AMUSE (Kim et al.
    2026, arXiv:2605.22432, eq. 4); AdamW runs without a first moment, as in SF-AdamW,
    since the averaging stands in for it. consistent_rms scales Muon's update to AdamW's
    RMS, so the two learning rates are in the same units.
    """
    warmup = lambda peak: optax.linear_schedule(0.0, peak, args.warmup_steps)
    base = optax.contrib.muon(
        warmup(args.learning_rate),
        beta=0.95,
        weight_decay=args.weight_decay,
        mu_dtype=jnp.bfloat16,
        consistent_rms=0.2,
        adam_b1=0.0,
        adam_b2=0.95,
        adam_learning_rate=warmup(args.adam_learning_rate),
        adam_weight_decay=args.weight_decay,
        muon_weight_dimension_numbers=muon_dimensions,
    )
    return optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.contrib.schedule_free(base, warmup(args.learning_rate), b1=0.9),
    )


def eval_flow(flow: LisaFlow, opt_state: optax.OptState) -> LisaFlow:
    """The averaged weights x that schedule-free evaluates, from the training weights y."""
    (state,) = [s for s in opt_state if isinstance(s, optax.contrib.ScheduleFreeState)]
    params, static = eqx.partition(flow, eqx.is_inexact_array)
    return eqx.combine(optax.contrib.schedule_free_eval_params(state, params), static)


def flow_loss(flow: LisaFlow, batch: TrainSample) -> Float[Array, ""]:
    du_pred = jax.vmap(flow)(batch.xt, batch.t, batch.y, batch.f)
    return jnp.mean(jnp.square(du_pred - batch.dx))


class TrainState(NamedTuple):
    problem: LisaGB
    flow: LisaFlow
    tx: optax.GradientTransformation
    opt_state: optax.OptState
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
            f_range=problem.f0_range,
            dtype=jnp.dtype(args.dtype),
            param_dtype=jnp.float32,
            key=key_network,
        )

        tx = optimizer(args)
        return cls(
            problem=problem,
            flow=flow,
            tx=tx,
            opt_state=tx.init(eqx.filter(flow, eqx.is_inexact_array)),
            key=key_train,
        )

    def routing_report(self) -> str:
        """How many parameters Muon and AdamW each take; Muon must have the backbone."""
        params = eqx.filter(self.flow, eqx.is_inexact_array)
        # Muon's size per parameter and AdamW's, labelled the way optax labels them
        is_dims = lambda d: d is None or isinstance(d, optax.contrib.MuonDimensionNumbers)
        split = lambda muon: jax.tree.map(
            lambda d, p: jax.tree.map(lambda a: a.size * ((d is not None) == muon), p),
            muon_dimensions(params),
            params,
            is_leaf=is_dims,
        )
        muon, adam = sum(jax.tree.leaves(split(True))), sum(jax.tree.leaves(split(False)))
        assert sum(jax.tree.leaves(split(False).backbone)) == 0, (
            "a backbone parameter is not on Muon"
        )
        return (
            f"[optimizer] Muon {muon / 1e6:.2f}M of {(muon + adam) / 1e6:.2f}M parameters"
            f" ({muon / (muon + adam):.1%}), AdamW the other {adam / 1e6:.2f}M"
        )

    @eqx.filter_jit
    def train_step(self, batch: TrainSample) -> tuple[Self, Float[Array, "2"]]:
        """One optimizer step on a batch; returns the flow loss and the gradient norm."""
        loss, grads = eqx.filter_value_and_grad(flow_loss)(self.flow, batch)
        params = eqx.filter(self.flow, eqx.is_inexact_array)
        updates, opt_state = self.tx.update(grads, self.opt_state, params)
        flow = eqx.apply_updates(self.flow, updates)
        state = self._replace(flow=flow, opt_state=opt_state)
        return state, jnp.stack([loss, optax.global_norm(grads)])

    @eqx.filter_jit
    def train_epoch(
        self, batch_size: int, n_steps: int, time_power: float = 1.0
    ) -> tuple[Self, Float[Array, "S 2"]]:
        """Fuse n_steps (gen + train_step) pairs into one XLA dispatch via lax.scan."""
        dynamic, static = eqx.partition(self, eqx.is_array)

        def scan_step(dynamic: Self, _) -> tuple[Self, Float[Array, "2"]]:
            state = eqx.combine(dynamic, static)
            key, key_batch = jr.split(state.key)
            batch = jax.vmap(partial(train_sample, state.problem, time_power=time_power))(
                jr.split(key_batch, batch_size)
            )
            state, losses = state._replace(key=key).train_step(batch)
            return eqx.filter(state, eqx.is_array), losses

        dynamic, losses = jax.lax.scan(scan_step, dynamic, length=n_steps)
        return eqx.combine(dynamic, static), losses

    @eqx.filter_jit
    def heldout_loss(
        self, key: Key[Array, ""], batch_size: int, time_power: float = 1.0
    ) -> Float[Array, ""]:
        """The flow loss of the averaged weights x on a batch drawn from `key`.

        The training loss is measured at y, where schedule-free takes its gradients; this is
        the model the eval and the scorecard sample from. The same key every epoch scores
        the same batch, so the curve is free of batch noise.
        """
        batch = jax.vmap(partial(train_sample, self.problem, time_power=time_power))(
            jr.split(key, batch_size)
        )
        return flow_loss(eval_flow(self.flow, self.opt_state), batch)

    def save_to(
        self,
        checkpoints: ocp.CheckpointManager,
        epoch: int,
        loss_hist: Float[Array, "E S 2"],
        heldout_hist: Float[Array, " E"],
    ) -> None:
        """Save the weights, the optimizer state and the histories as named checkpoint items.

        The optimizer state always goes in: schedule-free's averaged weights, the ones to
        evaluate, are rebuilt from the training weights and the z sequence it holds.
        """
        skip = ("problem", "tx", "key")
        checkpoints.save(
            epoch,
            args=ocp.args.Composite(
                loss_hist=ocp.args.ArraySave(loss_hist),
                heldout_hist=ocp.args.ArraySave(heldout_hist),
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
    ) -> tuple[Self, int, tuple[np.ndarray, np.ndarray] | None]:
        """Return the state, epoch and (loss, held-out) histories to resume from, or self at 0."""
        latest_epoch = checkpoints.latest_step()
        if latest_epoch is None:
            return self, 0, None

        skip = ("problem", "tx", "key")
        skeleton = {
            name: eqx.filter(field, eqx.is_array)
            for name, field in zip(self._fields, self)
            if name not in skip
        }
        # the histories and key are bare arrays, with no skeleton to take a sharding from,
        # so orbax would fall back to the one recorded at save time -- a device that need
        # not exist here, e.g. a GPU checkpoint read back on a CPU. Place them locally.
        local = ocp.ArrayRestoreArgs(
            sharding=jax.sharding.SingleDeviceSharding(jax.devices()[0])
        )
        restored = checkpoints.restore(
            latest_epoch,
            args=ocp.args.Composite(
                loss_hist=ocp.args.ArrayRestore(restore_args=local),
                heldout_hist=ocp.args.ArrayRestore(restore_args=local),
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
        print(f"[checkpoint] resuming at epoch {latest_epoch}", flush=True)
        return state, latest_epoch, (restored["loss_hist"], restored["heldout_hist"])


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

    The state's flow is the averaged weights x that schedule-free evaluates, unless
    args.weights is "training": then it is the training weights y. x is a uniform average
    over the whole run, so after a late drop in the loss it lags y (F33).

    For eval and scorecard. restore_from falls back to the fresh state when there is no
    checkpoint, which is right for a new training run but would quietly score an
    untrained network here, so stop instead.
    """
    out_dir = run_dir(args)
    checkpoints = checkpoint_manager(out_dir)
    state, epoch, _ = TrainState.from_config(args).restore_from(checkpoints)
    if epoch == 0:
        raise SystemExit(f"no checkpoint in {checkpoints.directory}: nothing to evaluate")
    # schedule-free trains at y and evaluates the average x
    if getattr(args, "weights", "average") == "training":
        return state, epoch, out_dir
    return state._replace(flow=eval_flow(state.flow, state.opt_state)), epoch, out_dir


def peak_memory_report() -> str:
    """The device's peak memory so far, or "" where the backend does not report it (the CPU)."""
    stats = jax.devices()[0].memory_stats() or {}
    if "peak_bytes_in_use" not in stats:
        return ""
    limit = stats.get("bytes_limit", 0) / 2**30
    return f"[memory] peak {stats['peak_bytes_in_use'] / 2**30:.1f} GiB of {limit:.0f} GiB"


def parse_args() -> argparse.Namespace:
    """Read the named run .yaml as argparse defaults, so any CLI flag overrides it."""
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default="B-fd", help="name of a run .yaml")
    config_parser.add_argument(
        "--config_root", type=Path, default=Path(__file__).parent / "configs"
    )
    config_args, _ = config_parser.parse_known_args()

    parser = argparse.ArgumentParser(parents=[config_parser])
    parser.add_argument("--output_dir", type=Path, default=Path("outputs"))
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Muon's")
    parser.add_argument("--adam_learning_rate", type=float, default=3e-4, help="AdamW's")
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=10_000,
        help="linear lr warmup from 0; schedule-free needs no other schedule",
    )
    parser.add_argument("--total_steps", type=int, default=500_000)
    parser.add_argument("--log_interval", type=int, default=1000)
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
    parser.add_argument(
        "--weights",
        default="average",
        choices=("average", "training"),
        help="eval and scorecard only: schedule-free's averaged weights x, or the training weights y",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=("bfloat16", "float32"),
        help="compute dtype for the network; params are kept in float32",
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
    print(state.routing_report(), flush=True)
    state, start_epoch, histories = state.restore_from(checkpoints)
    epochs = args.total_steps // args.log_interval
    # one fixed batch, never trained on, for the averaged weights' loss
    key_heldout = jr.fold_in(jr.key(args.seed), 1)

    # the checkpoint carries the (epochs, log_interval) grid it was written on, and
    # total_steps or log_interval may well have changed since -- extending a run is how
    # you spend more wall time -- so re-allocate at the current shape and copy the
    # overlap in, rather than indexing off the end of the restored array
    loss_history = np.full((epochs, args.log_interval, 2), np.nan)
    heldout_history = np.full((epochs,), np.nan)
    if histories is not None:
        restored, restored_heldout = histories
        rows = min(restored.shape[0], epochs)
        cols = min(restored.shape[1], args.log_interval)
        loss_history[:rows, :cols] = restored[:rows, :cols]
        heldout_history[:rows] = restored_heldout[:rows]

    pbar = tqdm(range(start_epoch, epochs), initial=start_epoch, total=epochs)
    for epoch in pbar:
        # one fused XLA dispatch for the whole epoch, instead of log_interval separate ones
        started = time.perf_counter()
        state, epoch_losses = state.train_epoch(
            args.batch_size, args.log_interval, args.time_power
        )
        loss_history[epoch] = jax.device_get(epoch_losses)
        heldout_history[epoch] = float(
            state.heldout_loss(key_heldout, args.batch_size, args.time_power)
        )
        seconds = time.perf_counter() - started  # the first epoch includes the compile

        # a non-finite loss has already poisoned the weights: stop, and leave the last
        # finite checkpoint in place instead of saving over it
        finite = np.isfinite(loss_history[epoch]).all(axis=-1)
        if not finite.all() or not np.isfinite(heldout_history[epoch]):
            checkpoints.close()
            raise SystemExit(
                f"[epoch {epoch + 1}] non-finite loss at step"
                f" {epoch * args.log_interval + int(np.argmin(finite))}, stopping"
            )

        # save a checkpoint and log the median of the epoch's losses
        state.save_to(checkpoints, epoch + 1, loss_history, heldout_history)
        flow_l, grad_norm = np.median(loss_history[epoch], axis=0)
        pbar.set_postfix(flow=f"{flow_l:.5f}", heldout=f"{heldout_history[epoch]:.5f}")
        print(
            f"[epoch {epoch + 1}/{epochs}] flow={flow_l:.5f}"
            f" heldout={heldout_history[epoch]:.5f} grad_norm={grad_norm:.3f}"
            f" time={seconds:.0f}s",
            flush=True,
        )
        if epoch == start_epoch and peak_memory_report():
            print(peak_memory_report(), flush=True)

        # redraw loss curve: median per epoch, shaded 10-90 percentile spread, and the
        # averaged weights' held-out loss
        xs = np.arange(1, epoch + 2) * args.log_interval
        l_lo, l_med, l_up = np.percentile(
            loss_history[: epoch + 1, :, 0], [10, 50, 90], axis=1
        )
        fig, ax = plt.subplots()
        ax.loglog(xs, l_med, label="flow (training weights y)", color="#2a86cf", lw=2)
        ax.fill_between(xs, l_lo, l_up, color="#2a86cf", alpha=0.15)
        ax.loglog(
            xs, heldout_history[: epoch + 1], label="held out (averaged weights x)",
            color="#d0622a", lw=2,
        )
        ax.set(xlabel="step", ylabel="loss", title=f"training losses ({run_id})")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(frameon=False)
        fig.savefig(os.path.join(out_dir, "losses.pdf"), bbox_inches="tight")
        plt.close(fig)

    # orbax saves in a background thread: the last one has to land before we exit
    checkpoints.close()
    print(f"[done] {run_id} -> {out_dir}", flush=True)
