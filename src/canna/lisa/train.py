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
from .injections import eval_injections
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


def match_f0(x0: Float[Array, "S 8"], x1: Float[Array, "S 8"]) -> Float[Array, "S 8"]:
    """x0's sources relabelled onto x1's by f0 rank: the k-th lowest f0 heads for the k-th lowest, so no paths cross in f0."""
    return x0[jnp.argsort(x0[:, 0])][jnp.argsort(jnp.argsort(x1[:, 0]))]


def train_sample(
    problem: LisaGB,
    key: Key[Array, ""],
    t: Float[Array, " K"] | None = None,
) -> TrainSample:
    """One observation (a window and its noisy data) and K points on straight paths to it, with their velocities x1 - x0.

    t: the K draws' times (stratified by the caller); one uniform draw when None.
    """
    key_c, key_p, key_o, key_x0, key_t = jr.split(key, 5)
    f = sample_f(problem, key_c)
    p = sample_physical(problem, key_p, f)
    y = preprocess(problem, sample_observation(problem, key_o, p, f), f)
    t = jr.uniform(key_t, (1,)) if t is None else t
    return paths(problem, key_x0, p, f, y, t)


def paths(
    problem: LisaGB,
    key: Key[Array, ""],
    p: Float[Array, "S 8"],
    f: Float[Array, ""],
    y: Float[Array, "F 6"],
    t: Float[Array, " K"],
) -> TrainSample:
    """K points at times t on straight paths from prior draws to the sources p of the observation y."""
    x0 = jax.vmap(sample_flow, in_axes=(None, 0, None))(
        problem, jr.split(key, t.shape[0]), f
    )
    x1 = physical_to_flow(problem, p, f)
    # the sources are exchangeable: each draw's sources head for x1's in f0 order
    x0 = jax.vmap(match_f0, in_axes=(0, None))(x0, x1)
    return TrainSample(
        xt=x0 + t[:, None, None] * (x1 - x0),
        dx=x1 - x0,
        t=t,
        y=y,
        x_target=x1,
        f=f,
    )


def losses(
    flow: LisaFlow,
    batch: TrainSample,
    by_bin: bool = False,
) -> Float[Array, "2"] | tuple[Float[Array, "2"], Float[Array, "3 T G"]]:
    """The flow's Gaussian NLL of x1, ((x1 - m) / e^s)² + 2 s, trained on as is, and the error of its mean m.

    by_bin: also, per (time bin, coordinate group), x1's squared error, the flow's σ² and their ratio.
    """
    # the data stream is encoded once per observation and shared by its K draws
    cache = jax.vmap(flow.encode)(batch.y, batch.f)
    per_draw = lambda f: jax.vmap(f, in_axes=(0, 0, None, None, None))
    m, s = jnp.split(
        jax.vmap(per_draw(flow))(batch.xt, batch.t, batch.y, batch.f, cache), 2, axis=-1
    )

    error = m - batch.x_target[:, None]
    flow_loss = jnp.mean(jnp.square(error * jnp.exp(-s)) + 2 * s)
    parts = jnp.stack([flow_loss, jnp.mean(jnp.square(error))])
    if not by_bin:
        return parts

    by_group = lambda a: jnp.stack(
        [jnp.mean(a[..., g], axis=(-2, -1)) for g in LOSS_GROUPS], axis=-1
    )
    decades = jnp.log10(jnp.maximum(1 - batch.t, 1e-30))
    bins = jnp.clip(
        (decades / LOSS_TIME_DECADES + 1) * LOSS_TIME_BINS, 0, LOSS_TIME_BINS - 1
    ).astype(int)
    per_draw_bins = jnp.stack(
        [
            by_group(jnp.square(error)),
            by_group(jnp.exp(2 * s)),
            by_group(jnp.square(error * jnp.exp(-s))),
        ]
    )
    one_hot = jax.nn.one_hot(bins, LOSS_TIME_BINS)
    binned = jnp.einsum("bkt,nbkg->ntg", one_hot, per_draw_bins)
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
    micro_batches: int = 1,
) -> tuple[TrainState, Float[Array, "3"]]:
    """One optimizer step on the flow loss; the flow loss, the x1 error and the grad norm."""

    def total(flow, chunk):
        parts = losses(flow, chunk)
        return parts[0], parts

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
    draws: int = 1,
    micro_batches: int = 1,
) -> tuple[TrainState, Float[Array, "S 3"]]:
    """Fuse n_steps (batch draw + train_step) pairs into one XLA dispatch via lax.scan.

    A batch is batch_size (x0, t) samples, `draws` per observation, t stratified over all of them.
    """
    observations = batch_size // draws
    dynamic, static = eqx.partition(state, eqx.is_array)
    sample = partial(train_sample, problem)

    def scan_step(dynamic: TrainState, _) -> tuple[TrainState, Float[Array, "3"]]:
        state = eqx.combine(dynamic, static)
        key, key_batch, key_perm, key_u = jr.split(state.key, 4)
        # one t in each of the batch's batch_size equal strata, shuffled
        u = (
            jr.permutation(key_perm, batch_size) + jr.uniform(key_u, (batch_size,))
        ) / batch_size
        batch = jax.vmap(sample)(
            jr.split(key_batch, observations), rearrange(u, "(b k) -> b k", k=draws)
        )
        state, parts = train_step(tx, state._replace(key=key), batch, micro_batches)
        return eqx.filter(state, eqx.is_array), parts

    dynamic, history = jax.lax.scan(scan_step, dynamic, length=n_steps)
    return eqx.combine(dynamic, static), history


class Catalog(NamedTuple):
    """The eval's injections and their observations, fixed for the run's diagnostics."""

    p: Float[Array, "J S 8"]
    f: Float[Array, " J"]
    y: Float[Array, "J F 6"]


def eval_catalog(problem: LisaGB, seed: int) -> Catalog:
    latents, windows, _, _, noise_keys, _ = eval_injections(problem, seed)
    observe = lambda key, p, f: preprocess(
        problem, sample_observation(problem, key, p, f), f
    )
    return Catalog(latents, windows, jax.vmap(observe)(noise_keys, latents, windows))


@eqx.filter_jit
def diagnostics(
    problem: LisaGB,
    state: TrainState,
    key: Key[Array, ""],
    catalog: Catalog,
) -> tuple[Float[Array, "4"], Float[Array, "3 T G"]]:
    """Flow loss of x and of y on the eval catalog, backbone weight RMS of x, and RMS of z - x.

    Also y's x1 error, expected error and their ratio per (time bin, coordinate group) on the catalog, one draw per
    injection in each bin.
    """
    key_batch, key_binned = jr.split(key)
    along = jax.vmap(partial(paths, problem), in_axes=(0, 0, 0, 0, None))
    n = catalog.f.shape[0]
    # LOSS_TIME_BINS times per injection: midpoints of equal steps in t, then of the log10(1 - t) bins
    u = (jnp.arange(LOSS_TIME_BINS) + 0.5) / LOSS_TIME_BINS
    batch = along(jr.split(key_batch, n), *catalog, u)
    binned = along(
        jr.split(key_binned, n), *catalog, 1 - 10.0 ** (-LOSS_TIME_DECADES * u)
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
    flow_loss = lambda flow: losses(flow, batch)[0]
    _, by_bin = losses(state.flow, binned, by_bin=True)
    return jnp.stack([flow_loss(x), flow_loss(state.flow), w_rms, spread]), by_bin


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


def warm_start(config: Config, flow: LisaFlow) -> LisaFlow:
    """flow with config.init_from's averaged weights at its latest checkpoint; the two runs must share the architecture, not the precision."""
    stored = yaml.safe_load(
        (config.output_dir / "runs" / config.init_from / "config.yaml").read_text()
    )
    name = stored.pop("name")
    source = Config(
        name=name, **(run_config.merge([stored]) | dict(output_dir=config.output_dir))
    )
    _, trained, epoch = load_trained(source)
    params = eqx.filter(flow, eqx.is_inexact_array)
    loaded = jax.tree.leaves(eqx.filter(trained, eqx.is_inexact_array))
    leaves, treedef = jax.tree.flatten(params)
    assert [a.shape for a in loaded] == [
        a.shape for a in leaves
    ], f"{config.init_from} has another architecture"
    copied = eqx.combine(jax.tree.unflatten(treedef, loaded), flow)
    print(f"[warm start] {config.init_from} epoch {epoch}", flush=True)
    return copied


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
        if git("rev-parse", "HEAD") != config.commit or git(
            "status", "--porcelain", "--", "."
        ):
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
            raise SystemExit(
                f"uncommitted changes in {src}: commit them to start a run"
            )
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
        loss_history=np.full((epochs, config.log_interval, 3), np.nan),
        catalog_history=np.full((epochs, 4), np.nan),
        bin_history=np.full((epochs, 3, LOSS_TIME_BINS, len(LOSS_GROUPS)), np.nan),
    )
    checkpoints = checkpoint_manager(run_dir)
    state, start_epoch, restored = restore_checkpoint(
        checkpoints, state, tuple(histories)
    )
    if start_epoch == 0 and config.init_from:
        flow = warm_start(config, state.flow)
        state = TrainState(
            flow, tx.init(eqx.filter(flow, eqx.is_inexact_array)), state.key
        )
    # the restored rows, into histories sized for this (maybe extended) run
    for name, rows in (restored or {}).items():
        histories[name][: len(rows)] = rows[: len(histories[name])]
    loss_history, catalog_history = (
        histories["loss_history"],
        histories["catalog_history"],
    )
    bin_history = histories["bin_history"]
    # the diagnostics' fixed draws: the eval's injections and noise, prior draws from a key the seed's setup leaves unused
    catalog = eval_catalog(problem, config.seed)
    key_catalog = jr.split(jr.key(config.seed), 3)[0]

    pbar = tqdm(range(start_epoch, epochs), initial=start_epoch, total=epochs)
    for epoch in pbar:
        started = time.perf_counter()
        state, epoch_losses = train_epoch(
            problem,
            tx,
            state,
            config.batch_size,
            config.log_interval,
            config.draws,
            config.micro_batches,
        )
        loss_history[epoch] = jax.device_get(epoch_losses)
        catalog_history[epoch], bin_history[epoch] = jax.device_get(
            diagnostics(
                problem,
                state,
                key_catalog,
                catalog,
            )
        )
        seconds = time.perf_counter() - started  # the first epoch includes the compile

        # a non-finite loss has poisoned the weights: stop before saving over the last checkpoint
        finite = np.isfinite(loss_history[epoch]).all(axis=-1)
        if not finite.all() or not np.isfinite(catalog_history[epoch]).all():
            checkpoints.close()
            where = (
                f"loss at step {epoch * config.log_interval + int(np.argmin(finite))}"
                if not finite.all()
                else "catalog loss"
            )
            raise SystemExit(f"[epoch {epoch + 1}] non-finite {where}, stopping")

        # orbax writes the checkpoint while the next epoch trains; log the median of the epoch's losses
        save_checkpoint(checkpoints, state, epoch + 1, histories)
        flow_l, x_l, grad_norm = np.median(loss_history[epoch], axis=0)
        catalog_x, catalog_y, w_rms, spread = catalog_history[epoch]
        pbar.set_postfix(flow=f"{flow_l:.5f}", catalog=f"{catalog_x:.5f}")
        print(
            f"[epoch {epoch + 1}/{epochs}] flow={flow_l:.5f} x={x_l:.5f}"
            f" catalog={catalog_x:.5f} catalog_y={catalog_y:.5f} grad_norm={grad_norm:.3f}"
            f" w_rms={w_rms:.4f} z_x={spread:.4f} time={seconds:.0f}s",
            flush=True,
        )
        if epoch == start_epoch:
            print_peak_memory()

        # the histories so far, for `-m canna.lisa.plots`; replaced whole so a reader never sees half a file
        done = epoch + 1
        np.savez(
            run_dir / "history.tmp.npz",
            epoch=done,
            steps=np.arange(1, done + 1) * config.log_interval,
            # float32: the plots take percentiles of them, and float64 doubles a long run's file
            losses=loss_history[:done].astype(np.float32),
            loss_names=np.array(["flow", "x", "grad_norm"]),
            catalog=catalog_history[:done],
            catalog_names=np.array(["catalog_x", "catalog_y", "w_rms", "z_x"]),
            bin_x_error=bin_history[:done, 0],
            bin_expected=bin_history[:done, 1],
            bin_ratio=bin_history[:done, 2],
            groups=np.array(["f0", "fdot", "sky", "z+", "z-"]),
            time_edges=LOSS_TIME_DECADES
            * (np.arange(LOSS_TIME_BINS + 1) / LOSS_TIME_BINS - 1),
        )
        os.replace(run_dir / "history.tmp.npz", run_dir / "history.npz")

    checkpoints.close()
    print(f"[done] {config.run} -> {run_dir}", flush=True)


if __name__ == "__main__":
    main()
