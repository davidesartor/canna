"""Training: the flow-matching draw, the losses, the scanned train epoch and the checkpoint."""

import dataclasses
import os
import subprocess
import time
from functools import partial
from pathlib import Path
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
import orbax.checkpoint as ocp
import yaml
from einops import rearrange
from jaxtyping import Array, Float, Key
from tqdm import tqdm

from . import config as run_config
from .chart import physical_to_flow, sample_flow
from .config import Config
from .network import (
    LOSS_GROUPS,
    LOSS_TIME_BINS,
    LOSS_TIME_DECADES,
    LisaFlow,
)
from .optim import eval_flow, optimizer, routing_report, schedule_free_state
from .problem import LisaGB, sample_f
from .simulator import preprocess, sample_observation, sample_physical


class TrainSample(NamedTuple):
    xt: Float[Array, "K S 8"]
    dx: Float[Array, "K S 8"]
    t: Float[Array, " K"]
    y: Float[Array, "F 6"]
    x_target: Float[Array, "S 8"]
    f: Float[Array, ""]


def train_sample(
    problem: LisaGB,
    key: Key[Array, ""],
    u: Float[Array, " K"] | None = None,
    time_power: float = 1.0,
    late_time_mix: float = 0.0,
    late_time_floor: float = 1e-4,
) -> TrainSample:
    """One observation (a window and its noisy data) and K points on straight paths to it, with their velocities d/ds.

    u: the K draws' uniforms for t (stratified by the caller); one fresh draw when None.
    """
    key_c, key_p, key_o, key_x0, key_t = jr.split(key, 5)
    f = sample_f(problem, key_c)
    p = sample_physical(problem, key_p, f)
    y = preprocess(problem, sample_observation(problem, key_o, p, f), f)

    # t is uniform, the point sits at s = 1 - (1 - t)^time_power along the path, the target is d/ds there
    t = jr.uniform(key_t, (1,)) if u is None else u
    x0 = jax.vmap(sample_flow, in_axes=(None, 0, None))(
        problem, jr.split(key_x0, t.shape[0]), f
    )
    x1 = physical_to_flow(problem, p, f)
    # a late_time_mix share of draws has 1 - s log-uniform on [late_time_floor, 1] instead
    if late_time_mix > 0:
        key_mix = jr.fold_in(key_t, 1)
        t_late = 1 - late_time_floor ** (t / time_power)
        t = jnp.where(jr.bernoulli(key_mix, late_time_mix, t.shape), t_late, t)
    s = t if time_power == 1 else 1 - (1 - t) ** time_power
    return TrainSample(
        xt=x0 + s[:, None, None] * (x1 - x0),
        dx=x1 - x0,
        t=t,
        y=y,
        x_target=x1,
        f=f,
    )


def losses(
    flow: LisaFlow,
    batch: TrainSample,
    time_power: float = 1.0,
    use_hat: bool | Array = False,
    by_bin: bool = False,
) -> Float[Array, "3"] | tuple[Float[Array, "3"], Float[Array, "T G"]]:
    """The flow-matching loss, the error of the x1 it implies, and the flow objective trained on.

    by_bin: also the mean squared error per (time bin, coordinate group) the learned weights see.
    """
    # the data stream is encoded once per observation and shared by its K draws
    cache = jax.vmap(flow.encode)(batch.y, batch.f)
    remaining = ((1 - batch.t) ** time_power)[..., None, None]
    per_draw = lambda f: jax.vmap(f, in_axes=(0, 0, None, None, 0, None))
    per_observation = jax.vmap(per_draw(flow))

    def predict(x_hat):
        out = per_observation(batch.xt, batch.t, batch.y, batch.f, x_hat, cache)
        x1 = out if flow.prediction == "x" else batch.xt + remaining * out
        return out, x1

    # self-conditioning: on use_hat, a first, stopped pass's x1 conditions the second
    if flow.self_condition:
        first = lambda: jax.lax.stop_gradient(predict(None)[1])
        out, x_pred = jax.lax.cond(
            use_hat, lambda: predict(first()), lambda: predict(None)
        )
    else:
        out, x_pred = predict(None)

    # x-prediction: the velocity (x1 - xt) / (1 - s), floored at the last λ bin's 1 - s, below which no weight can tame it
    if flow.prediction == "x":
        error = (out - batch.xt - remaining * batch.dx) / jnp.maximum(
            remaining, 10.0**-LOSS_TIME_DECADES
        )
    else:
        error = out - batch.dx
    flow_loss = jnp.mean(jnp.square(error))

    # learned loss weights (Kendall et al. 2018, EDM2): e exp(-lambda) + lambda per (time bin, coordinate group)
    groups = jnp.stack(
        [jnp.mean(jnp.square(error[..., g]), axis=(-2, -1)) for g in LOSS_GROUPS],
        axis=-1,
    )
    decades = time_power * jnp.log10(jnp.maximum(1 - batch.t, 1e-30))
    bins = jnp.clip(
        (decades / LOSS_TIME_DECADES + 1) * LOSS_TIME_BINS, 0, LOSS_TIME_BINS - 1
    ).astype(int)
    objective = flow_loss
    if flow.loss_log_var is not None:
        log_var = flow.loss_log_var[bins]
        objective = jnp.mean(groups * jnp.exp(-log_var) + log_var)
    x_error = jnp.mean(jnp.square(x_pred - batch.x_target[:, None]))
    parts = jnp.stack([flow_loss, x_error, objective])
    if not by_bin:
        return parts
    one_hot = jax.nn.one_hot(bins, LOSS_TIME_BINS)
    binned = jnp.einsum("bkt,bkg->tg", one_hot, groups)
    count = jnp.sum(one_hot, axis=(0, 1))[:, None]
    # an empty bin has no error to report
    return parts, jnp.where(count > 0, binned / jnp.maximum(count, 1), jnp.nan)


class TrainState(NamedTuple):
    flow: LisaFlow
    opt_state: optax.OptState
    key: Key[Array, ""]


def setup(
    config: Config,
) -> tuple[LisaGB, optax.GradientTransformation, TrainState]:
    """The problem, the optimizer and a fresh training state, all from the run config."""
    _, key_network, key_train = jr.split(jr.key(config.seed), 3)
    problem = LisaGB(**config.problem)
    flow = LisaFlow(
        problem,
        **config.network,
        dtype=jnp.bfloat16 if config.bf16 == "backbone" else jnp.float32,
        attention_dtype=None if config.bf16 == "none" else jnp.bfloat16,
        param_dtype=jnp.float32,
        key=key_network,
    )
    tx = optimizer(config)
    opt_state = tx.init(eqx.filter(flow, eqx.is_inexact_array))
    return problem, tx, TrainState(flow, opt_state, key_train)


@eqx.filter_jit
def train_step(
    tx: optax.GradientTransformation,
    state: TrainState,
    batch: TrainSample,
    time_power: float = 1.0,
    use_hat: bool | Array = False,
    micro_batches: int = 1,
) -> tuple[TrainState, Float[Array, "4"]]:
    """One optimizer step on the flow objective; the flow loss, the x1 error, the objective and the grad norm."""

    def total(flow, chunk):
        parts = losses(flow, chunk, time_power, use_hat)
        return parts[2], parts

    # the batch's gradient as the mean over equal chunks, one chunk's activations live at a time
    grad = eqx.filter_value_and_grad(total, has_aux=True)
    params = eqx.filter(state.flow, eqx.is_inexact_array)

    def chunk_grad(chunk):
        (_, parts), grads = grad(state.flow, chunk)
        return parts, grads

    def accumulate(carry, chunk):
        return jax.tree.map(jnp.add, carry, chunk_grad(chunk)), None

    chunks = jax.tree.map(
        lambda a: rearrange(a, "(m b) ... -> m b ...", m=micro_batches), batch
    )
    zeros = jax.tree.map(
        lambda a: jnp.zeros(a.shape, a.dtype),
        jax.eval_shape(chunk_grad, jax.tree.map(lambda a: a[0], chunks)),
    )
    (parts, grads), _ = jax.lax.scan(accumulate, zeros, chunks)
    parts, grads = jax.tree.map(lambda a: a / micro_batches, (parts, grads))
    updates, opt_state = tx.update(grads, state.opt_state, params)
    state = state._replace(
        flow=eqx.apply_updates(state.flow, updates), opt_state=opt_state
    )
    return state, jnp.concat([parts, optax.global_norm(grads)[None]])


@eqx.filter_jit
def train_epoch(
    problem: LisaGB,
    tx: optax.GradientTransformation,
    state: TrainState,
    batch_size: int,
    n_steps: int,
    time_power: float = 1.0,
    draws: int = 1,
    late_time_mix: float = 0.0,
    late_time_floor: float = 1e-4,
    micro_batches: int = 1,
) -> tuple[TrainState, Float[Array, "S 4"]]:
    """Fuse n_steps (batch draw + train_step) pairs into one XLA dispatch via lax.scan.

    A batch is batch_size (x0, t) samples, `draws` per observation, t stratified over all of them.
    """
    observations = batch_size // draws
    dynamic, static = eqx.partition(state, eqx.is_array)
    sample = partial(
        train_sample,
        problem,
        time_power=time_power,
        late_time_mix=late_time_mix,
        late_time_floor=late_time_floor,
    )

    def scan_step(dynamic: TrainState, _) -> tuple[TrainState, Float[Array, "4"]]:
        state = eqx.combine(dynamic, static)
        key, key_batch, key_perm, key_u, key_hat = jr.split(state.key, 5)
        # one t in each of the batch's batch_size equal strata, shuffled
        u = (
            jr.permutation(key_perm, batch_size) + jr.uniform(key_u, (batch_size,))
        ) / batch_size
        batch = jax.vmap(sample)(
            jr.split(key_batch, observations), rearrange(u, "(b k) -> b k", k=draws)
        )
        # half the steps self-condition (when the flow does)
        use_hat = jr.bernoulli(key_hat)
        state, parts = train_step(
            tx, state._replace(key=key), batch, time_power, use_hat, micro_batches
        )
        return eqx.filter(state, eqx.is_array), parts

    dynamic, history = jax.lax.scan(scan_step, dynamic, length=n_steps)
    return eqx.combine(dynamic, static), history


@eqx.filter_jit
def diagnostics(
    problem: LisaGB,
    state: TrainState,
    key: Key[Array, ""],
    batch_size: int,
    time_power: float = 1.0,
) -> tuple[Float[Array, "4"], Float[Array, "T G"]]:
    """Flow loss (self-conditioned, as sampled) of x and of y, backbone weight RMS of x, and RMS of z - x, on the batch from `key`.

    Also y's error per (time bin, coordinate group), on a second batch whose 1 - s is stratified log-uniformly over the bins.
    """
    key_batch, key_binned, key_u = jr.split(key, 3)
    sample = partial(train_sample, problem, time_power=time_power)
    batch = jax.vmap(sample)(jr.split(key_batch, batch_size))
    # every draw late, 1 - s = floor^u: batch_size / LOSS_TIME_BINS draws in each bin
    spread_over_bins = partial(
        sample, late_time_mix=1.0, late_time_floor=10.0**-LOSS_TIME_DECADES
    )
    u = (jnp.arange(batch_size) + jr.uniform(key_u, (batch_size,))) / batch_size
    binned = jax.vmap(spread_over_bins)(
        jr.split(key_binned, batch_size), rearrange(u, "b -> b 1")
    )
    x = eval_flow(state.flow, state.opt_state)
    rms = lambda tree: jnp.sqrt(
        sum(jnp.sum(jnp.square(a.astype(jnp.float32))) for a in jax.tree.leaves(tree))
        / sum(a.size for a in jax.tree.leaves(tree))
    )
    w_rms = rms(eqx.filter(x.backbone, eqx.is_inexact_array))
    z = schedule_free_state(state.opt_state).z
    spread = rms(
        jax.tree.map(lambda zi, xi: zi - xi, z, eqx.filter(x, eqx.is_inexact_array))
    )
    held_out = lambda flow: losses(flow, batch, time_power, True)[0]
    _, bin_error = losses(state.flow, binned, time_power, True, by_bin=True)
    return jnp.stack([held_out(x), held_out(state.flow), w_rms, spread]), bin_error


def checkpoint_manager(run_dir: Path, read_only: bool = False) -> ocp.CheckpointManager:
    """The run's checkpoints, keeping only the latest (orbax holds the old one while saving).

    read_only: for a reader beside a live trainer, which must not delete the save it is writing.
    """
    # a requeue can kill a save midway: drop its partial folder when the run restarts
    return ocp.CheckpointManager(
        (run_dir / "checkpoints").absolute(),
        options=ocp.CheckpointManagerOptions(
            max_to_keep=None if read_only else 1,
            cleanup_tmp_directories=not read_only,
            read_only=read_only,
        ),
    )


def save_checkpoint(
    checkpoints: ocp.CheckpointManager,
    state: TrainState,
    epoch: int,
    histories: dict[str, np.ndarray],
) -> None:
    """Save the training weights, the optimizer state (it holds the averaged x), the key and the named histories; orbax writes them in the background."""
    checkpoints.save(
        epoch,
        args=ocp.args.Composite(
            flow=ocp.args.StandardSave(eqx.filter(state.flow, eqx.is_array)),
            opt_state=ocp.args.StandardSave(eqx.filter(state.opt_state, eqx.is_array)),
            # a bare key array is not a pytree StandardSave will take
            key=ocp.args.ArraySave(jr.key_data(state.key)),
            # copies: the trainer fills the next epoch's rows while this one is written
            **{name: ocp.args.ArraySave(h.copy()) for name, h in histories.items()},
        ),
    )


def restore_checkpoint(
    checkpoints: ocp.CheckpointManager,
    state: TrainState,
    histories: tuple[str, ...] = (),
) -> tuple[TrainState, int, dict[str, np.ndarray] | None]:
    """The state, epoch and the named histories to resume from, or `state` at 0."""
    epoch = checkpoints.latest_step()
    if epoch is None:
        return state, 0, None

    # the key and histories have no skeleton to take a sharding from, so orbax would fall back
    # to the one recorded at save time -- a device that need not exist here (a GPU checkpoint
    # read on a CPU). Place them locally.
    local = ocp.ArrayRestoreArgs(
        sharding=jax.sharding.SingleDeviceSharding(jax.devices()[0])
    )
    restored = checkpoints.restore(
        epoch,
        args=ocp.args.Composite(
            flow=ocp.args.StandardRestore(eqx.filter(state.flow, eqx.is_array)),
            opt_state=ocp.args.StandardRestore(
                eqx.filter(state.opt_state, eqx.is_array)
            ),
            key=ocp.args.ArrayRestore(restore_args=local),
            **{name: ocp.args.ArrayRestore(restore_args=local) for name in histories},
        ),
    )
    state = TrainState(
        flow=eqx.combine(restored["flow"], state.flow),
        opt_state=eqx.combine(restored["opt_state"], state.opt_state),
        key=jr.wrap_key_data(restored["key"]),
    )
    print(f"[checkpoint] resuming at epoch {epoch}", flush=True)
    return state, epoch, {name: np.asarray(restored[name]) for name in histories}


def load_trained(config: Config) -> tuple[LisaGB, LisaFlow, int]:
    """The problem, the trained flow (averaged x, or y with weights: training) and its epoch."""
    if not (config.run_dir / "checkpoints").is_dir():
        raise SystemExit(f"no checkpoint in {config.run_dir}: nothing to evaluate")
    problem, _, skeleton = setup(config)
    with checkpoint_manager(config.run_dir, read_only=True) as checkpoints:
        state, epoch, _ = restore_checkpoint(checkpoints, skeleton)
    if epoch == 0:
        raise SystemExit(f"no checkpoint in {config.run_dir}: nothing to evaluate")
    if config.weights == "training":
        return problem, state.flow, epoch
    return problem, eval_flow(state.flow, state.opt_state), epoch


def print_peak_memory() -> None:
    """The device's peak memory so far, where the backend reports it (not the CPU)."""
    stats = jax.devices()[0].memory_stats() or {}
    if "peak_bytes_in_use" in stats:
        print(
            f"[memory] peak {stats['peak_bytes_in_use'] / 2**30:.1f} GiB"
            f" of {stats.get('bytes_limit', 0) / 2**30:.0f} GiB",
            flush=True,
        )


def main() -> None:
    config = run_config.load()
    src = Path(__file__).parents[2]
    git = lambda *args: subprocess.run(
        ["git", "-C", str(src), *args], capture_output=True, text=True, check=True
    ).stdout.strip()
    if config.run and (config.run_dir / "config.yaml").exists():
        # a resume: the run's config.yaml holds, on the source it trained at
        config = run_config.load_run(allowed=run_config.RESUME_FIELDS)
        if git("rev-parse", "HEAD") != config.commit or git("status", "--porcelain", "--", "."):
            raise SystemExit(
                f"{config.run} trained at {config.commit}: resume it from that commit's"
                f" worktree, outputs/src/{config.commit}"
            )
        # an extended (or shortened) run: its config.yaml records the new length
        record = config.run_dir / "config.yaml"
        stored = yaml.safe_load(record.read_text())
        if stored["total_steps"] != config.total_steps:
            stored["total_steps"] = config.total_steps
            record.write_text(yaml.safe_dump(stored, sort_keys=False))
    else:
        # a new run: pinned to a commit, so its source must be committed
        if git("status", "--porcelain", "--", "."):
            raise SystemExit(f"uncommitted changes in {src}: commit them to start a run")
        config = dataclasses.replace(
            config,
            run=config.run or run_config.run_name(config.name, config.output_dir),
            commit=git("rev-parse", "HEAD"),
        )
        config.run_dir.mkdir(parents=True, exist_ok=True)
        run_config.save(config, config.run_dir / "config.yaml")
    run_dir = config.run_dir
    print(f"[run] {config.run} at {config.commit} -> {run_dir}", flush=True)
    print(f"JAX backend: {jax.default_backend()}", flush=True)
    print(f"devices: {jax.local_device_count()}", flush=True)

    problem, tx, state = setup(config)
    print(routing_report(state.flow), flush=True)
    epochs = config.total_steps // config.log_interval
    histories = dict(
        loss_history=np.full((epochs, config.log_interval, 4), np.nan),
        heldout_history=np.full((epochs, 4), np.nan),
        log_var_history=np.full((epochs, LOSS_TIME_BINS, len(LOSS_GROUPS)), np.nan),
        bin_error_history=np.full((epochs, LOSS_TIME_BINS, len(LOSS_GROUPS)), np.nan),
    )
    checkpoints = checkpoint_manager(run_dir)
    state, start_epoch, restored = restore_checkpoint(checkpoints, state, tuple(histories))
    # the restored rows, into histories sized for this (maybe extended) run
    for name, rows in (restored or {}).items():
        histories[name][: len(rows)] = rows[: len(histories[name])]
    loss_history, heldout_history = histories["loss_history"], histories["heldout_history"]
    log_var_history = histories["log_var_history"]
    bin_error_history = histories["bin_error_history"]
    # one fixed batch, never trained on, for the diagnostics: the seed's key setup leaves unused
    key_heldout = jr.split(jr.key(config.seed), 3)[0]

    pbar = tqdm(range(start_epoch, epochs), initial=start_epoch, total=epochs)
    for epoch in pbar:
        started = time.perf_counter()
        state, epoch_losses = train_epoch(
            problem,
            tx,
            state,
            config.batch_size,
            config.log_interval,
            config.time_power,
            config.draws,
            config.late_time_mix,
            config.late_time_floor,
            config.micro_batches,
        )
        loss_history[epoch] = jax.device_get(epoch_losses)
        heldout_history[epoch], bin_error_history[epoch] = jax.device_get(
            diagnostics(
                problem,
                state,
                key_heldout,
                config.batch_size // config.draws,
                config.time_power,
            )
        )
        if state.flow.loss_log_var is not None:
            log_var_history[epoch] = jax.device_get(state.flow.loss_log_var)
        seconds = time.perf_counter() - started  # the first epoch includes the compile

        # a non-finite loss has poisoned the weights: stop before saving over the last checkpoint
        finite = np.isfinite(loss_history[epoch]).all(axis=-1)
        if not finite.all() or not np.isfinite(heldout_history[epoch]).all():
            checkpoints.close()
            raise SystemExit(
                f"[epoch {epoch + 1}] non-finite loss at step"
                f" {epoch * config.log_interval + int(np.argmin(finite))}, stopping"
            )

        # orbax writes the checkpoint while the next epoch trains; log the median of the epoch's losses
        save_checkpoint(checkpoints, state, epoch + 1, histories)
        flow_l, x_l, _, grad_norm = np.median(loss_history[epoch], axis=0)
        held_x, held_y, w_rms, spread = heldout_history[epoch]
        pbar.set_postfix(flow=f"{flow_l:.5f}", heldout=f"{held_x:.5f}")
        print(
            f"[epoch {epoch + 1}/{epochs}] flow={flow_l:.5f} x={x_l:.5f}"
            f" heldout={held_x:.5f} heldout_y={held_y:.5f} grad_norm={grad_norm:.3f}"
            f" w_rms={w_rms:.4f} z_x={spread:.4f} time={seconds:.0f}s",
            flush=True,
        )
        if epoch == start_epoch:
            print_peak_memory()

        # the histories so far, for `-m canna.lisa.plots`; replaced whole so a reader never sees half a file
        done = epoch + 1
        log_var = log_var_history[:done]
        np.savez(
            run_dir / "history.tmp.npz",
            epoch=done,
            steps=np.arange(1, done + 1) * config.log_interval,
            # float32: the plots take percentiles of them, and float64 doubles a long run's file
            losses=loss_history[:done].astype(np.float32),
            loss_names=np.array(["flow", "x", "objective", "grad_norm"]),
            heldout=heldout_history[:done],
            heldout_names=np.array(["heldout_x", "heldout_y", "w_rms", "z_x"]),
            log_var=log_var,
            bin_error=bin_error_history[:done],
            bin_objective=np.where(
                np.isnan(log_var),
                bin_error_history[:done],
                bin_error_history[:done] * np.exp(-log_var) + log_var,
            ),
            groups=np.array(["f0", "Mc", "sky", "z+", "z-"]),
            time_edges=LOSS_TIME_DECADES * (np.arange(LOSS_TIME_BINS + 1) / LOSS_TIME_BINS - 1),
        )
        os.replace(run_dir / "history.tmp.npz", run_dir / "history.npz")

    checkpoints.close()
    print(f"[done] {config.run} -> {run_dir}", flush=True)


if __name__ == "__main__":
    main()
