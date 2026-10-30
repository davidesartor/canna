"""Training: the flow-matching draw, the losses, the scanned train epoch and the checkpoint."""

import dataclasses
import math
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
from .chart import physical_to_flow
from .config import Config
from .injections import eval_injections
from .network import (
    LOSS_GROUPS,
    LOSS_LOG10_SNR,
    LOSS_SNR_BINS,
    LOSS_TIME_BINS,
    LOSS_TIME_DECADES,
    LisaFlow,
    reference_frame,
)
from .optim import eval_flow, optimizer, routing_report, schedule_free_state
from .problem import LisaGB, f0_window
from .simulator import (
    preprocess,
    sample_observation,
    sample_physical,
    sample_window,
    source_snrs,
)

# the frame loss's ridge on (delta_mu, delta_s), in prior units
FRAME_DECAY = 0.01


class TrainSample(NamedTuple):
    z0: Float[Array, "K S 8"]
    t: Float[Array, " K"]
    y: Float[Array, "F 6"]
    x1: Float[Array, "S 8"]
    f: Float[Array, ""]


def train_sample(
    problem: LisaGB,
    key: Key[Array, ""],
    t: Float[Array, " K"] | None = None,
) -> TrainSample:
    """One observation (a window and its noisy data), its sources x1 in f0 order, and K base draws at times t.

    t: the K draws' times (stratified by the caller); one uniform draw when None.
    """
    key_c, key_p, key_o, key_z0, key_t = jr.split(key, 5)
    f = sample_window(problem, key_c)
    p = sample_physical(problem, key_p, f)
    y = preprocess(problem, sample_observation(problem, key_o, p, f), f)
    t = jr.uniform(key_t, (1,)) if t is None else t
    return paths(problem, key_z0, p, f, y, t)


def paths(
    problem: LisaGB,
    key: Key[Array, ""],
    p: Float[Array, "S 8"],
    f: Float[Array, ""],
    y: Float[Array, "F 6"],
    t: Float[Array, " K"],
) -> TrainSample:
    """The sources p of the observation y as x1, in f0 order, and K standard normal base draws z0 at times t."""
    p = p[jnp.argsort(p[:, 0])]
    return TrainSample(
        z0=jr.normal(key, (t.shape[0], *p.shape)),
        t=t,
        y=y,
        x1=physical_to_flow(problem, p, f),
        f=f,
    )


class Binned(NamedTuple):
    """The diagnostics per bin: the frame per (snr bin, group), the flow per (time bin, group)."""

    # mean log10(sigma_A / sigma_prior) and mean k_A²
    frame: Float[Array, "2 N G"]
    # mean log10 of f0's sigma_A over the monochromatic Fisher width sqrt(3) / (pi snr), in bins
    f0_fisher: Float[Array, " N"]
    # mean log10 sigma_z and mean k_z²
    flow: Float[Array, "2 T G"]


def losses(
    flow: LisaFlow,
    problem: LisaGB,
    batch: TrainSample,
    snr: Float[Array, "B S"] | None = None,
) -> Float[Array, "3"] | tuple[Float[Array, "3"], Binned]:
    """The objective, the frame's NLL of x1 and the flow's NLL of z1, all trained on but the frame NLL as is.

    The frame (mu_A, sigma_A) = (mu_p + sigma_p delta_mu, sigma_p e^{delta_s}) is fit by the NLL of x1 weighted by
    sg(e^{delta_s}) with a ridge on delta; the flow by the Gaussian NLL of z1 = sg((x1 - mu_A) / sigma_A) on paths
    from z0. snr: each source's own, for the diagnostics per bin.
    """
    # the data stream is encoded once per observation and shared by its K draws
    delta, cache = jax.vmap(flow.encode)(batch.y, batch.f)
    frame = jax.vmap(partial(reference_frame, problem))(delta, batch.f)
    mu, log_sigma = frame
    delta_mu, delta_s = jnp.split(delta, 2, axis=-1)
    k = (batch.x1 - mu) * jnp.exp(-log_sigma)
    frame_nll = jnp.square(k) + 2 * delta_s
    frame_loss = jnp.mean(
        jax.lax.stop_gradient(jnp.exp(delta_s))
        * (frame_nll + FRAME_DECAY * (jnp.square(delta_mu) + jnp.square(delta_s)))
    )

    z1 = jax.lax.stop_gradient(k)[:, None]
    t = batch.t[..., None, None]
    zt = (1 - t) * batch.z0 + t * z1
    per_draw = lambda f: jax.vmap(f, in_axes=(0, 0, None, None, None))
    m, s = jnp.split(
        jax.vmap(per_draw(flow))(zt, batch.t, frame, batch.f, cache), 2, axis=-1
    )
    k_z = (z1 - m) * jnp.exp(-s)
    flow_nll = jnp.mean(jnp.square(k_z) + 2 * s)
    parts = jnp.stack([frame_loss + flow_nll, jnp.mean(frame_nll), flow_nll])
    if snr is None:
        return parts

    by_group = lambda a: jnp.stack(
        [jnp.mean(a[..., g], axis=-1) for g in LOSS_GROUPS], axis=-1
    )

    def binned_mean(pattern: str, bins: Array, n_bins: int, a: Array) -> Array:
        """a's mean per bin, the bin axis last; an empty bin has nothing to report."""
        one_hot = jax.nn.one_hot(bins, n_bins)
        count = jnp.sum(one_hot, axis=(0, 1))
        return jnp.einsum(pattern, one_hot, a) / jnp.where(count > 0, count, jnp.nan)

    # the frame per source, by its own true snr
    snr_bins = jnp.clip(
        (jnp.log10(snr) - LOSS_LOG10_SNR[0])
        / (LOSS_LOG10_SNR[1] - LOSS_LOG10_SNR[0])
        * LOSS_SNR_BINS,
        0,
        LOSS_SNR_BINS - 1,
    ).astype(int)
    frame_bins = binned_mean(
        "bsn,ibsg->ign",
        snr_bins,
        LOSS_SNR_BINS,
        jnp.stack([by_group(delta_s / math.log(10)), by_group(jnp.square(k))]),
    )
    # f0's sigma_A in bins: x_f maps log f0 onto [-1, 1] over the window's f0 range
    lo, hi = jax.vmap(partial(f0_window, problem))(batch.f)
    half_log_span = 0.5 * jnp.log(hi / lo)[:, None]
    f0 = jnp.sqrt(lo * hi)[:, None] * jnp.exp(batch.x1[..., 0] * half_log_span)
    sigma_bins = jnp.exp(log_sigma[..., 0]) * half_log_span * f0 * problem.t_obs
    fisher = jnp.log10(sigma_bins * jnp.pi * snr / math.sqrt(3))

    # the flow per draw, by its log10(1 - t)
    decades = jnp.log10(jnp.maximum(1 - batch.t, 1e-30))
    time_bins = jnp.clip(
        (decades / LOSS_TIME_DECADES + 1) * LOSS_TIME_BINS, 0, LOSS_TIME_BINS - 1
    ).astype(int)
    flow_bins = binned_mean(
        "bkt,ibkg->igt",
        time_bins,
        LOSS_TIME_BINS,
        jnp.stack(
            [
                jnp.mean(by_group(s / math.log(10)), axis=-2),
                jnp.mean(by_group(jnp.square(k_z)), axis=-2),
            ]
        ),
    )
    return parts, Binned(
        frame=rearrange(frame_bins, "i g n -> i n g"),
        f0_fisher=binned_mean("bsn,bs->n", snr_bins, LOSS_SNR_BINS, fisher),
        flow=rearrange(flow_bins, "i g t -> i t g"),
    )


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
    problem: LisaGB,
    tx: optax.GradientTransformation,
    state: TrainState,
    batch: TrainSample,
    micro_batches: int = 1,
) -> tuple[TrainState, Float[Array, "3"]]:
    """One optimizer step on the objective; the frame NLL, the flow NLL and the grad norm."""

    def total(flow, chunk):
        parts = losses(flow, problem, chunk)
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
    return state, jnp.concat([parts[1:], optax.global_norm(grads)[None]])


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

    A batch is batch_size (z0, t) samples, `draws` per observation, t stratified over all of them.
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
        state, parts = train_step(
            problem, tx, state._replace(key=key), batch, micro_batches
        )
        return eqx.filter(state, eqx.is_array), parts

    dynamic, history = jax.lax.scan(scan_step, dynamic, length=n_steps)
    return eqx.combine(dynamic, static), history


class Catalog(NamedTuple):
    """The eval's injections in f0 order, their observations and each source's own snr, fixed for the run's diagnostics."""

    p: Float[Array, "J S 8"]
    f: Float[Array, " J"]
    y: Float[Array, "J F 6"]
    snr: Float[Array, "J S"]


def eval_catalog(problem: LisaGB, seed: int) -> Catalog:
    latents, windows, _, _, noise_keys, _ = eval_injections(problem, seed)
    observe = lambda key, p, f: preprocess(
        problem, sample_observation(problem, key, p, f), f
    )
    latents = jnp.take_along_axis(
        latents, jnp.argsort(latents[..., 0], axis=-1)[..., None], axis=-2
    )
    return Catalog(
        latents,
        windows,
        jax.vmap(observe)(noise_keys, latents, windows),
        jax.vmap(partial(source_snrs, problem))(latents, windows),
    )


@eqx.filter_jit
def diagnostics(
    problem: LisaGB,
    state: TrainState,
    key: Key[Array, ""],
    catalog: Catalog,
) -> tuple[Float[Array, "6"], Binned]:
    """Frame and flow NLL of x and of y on the eval catalog, backbone weight RMS of x, and RMS of z - x.

    Also y's frame per (snr bin, group) and flow per (time bin, group) on the catalog, one draw per injection in each
    time bin.
    """
    key_batch, key_binned = jr.split(key)
    along = jax.vmap(partial(paths, problem), in_axes=(0, 0, 0, 0, None))
    n = catalog.f.shape[0]
    # LOSS_TIME_BINS times per injection: midpoints of equal steps in t, then of the log10(1 - t) bins
    u = (jnp.arange(LOSS_TIME_BINS) + 0.5) / LOSS_TIME_BINS
    batch = along(jr.split(key_batch, n), catalog.p, catalog.f, catalog.y, u)
    binned = along(
        jr.split(key_binned, n),
        catalog.p,
        catalog.f,
        catalog.y,
        1 - 10.0 ** (-LOSS_TIME_DECADES * u),
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
    nll = lambda flow: losses(flow, problem, batch)[1:]
    _, by_bin = losses(state.flow, problem, binned, catalog.snr)
    return jnp.concat([nll(x), nll(state.flow), jnp.stack([w_rms, spread])]), by_bin


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
        catalog_history=np.full((epochs, 6), np.nan),
        frame_history=np.full((epochs, 2, LOSS_SNR_BINS, len(LOSS_GROUPS)), np.nan),
        fisher_history=np.full((epochs, LOSS_SNR_BINS), np.nan),
        flow_history=np.full((epochs, 2, LOSS_TIME_BINS, len(LOSS_GROUPS)), np.nan),
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
    frame_history, fisher_history, flow_history = (
        histories["frame_history"],
        histories["fisher_history"],
        histories["flow_history"],
    )
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
        catalog_history[epoch], binned = jax.device_get(
            diagnostics(problem, state, key_catalog, catalog)
        )
        frame_history[epoch], fisher_history[epoch], flow_history[epoch] = binned
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
        frame_l, flow_l, grad_norm = np.median(loss_history[epoch], axis=0)
        frame_x, flow_x, frame_y, flow_y, w_rms, spread = catalog_history[epoch]
        pbar.set_postfix(frame=f"{frame_x:.4f}", flow=f"{flow_x:.4f}")
        print(
            f"[epoch {epoch + 1}/{epochs}] frame={frame_l:.4f} flow={flow_l:.4f}"
            f" catalog frame={frame_x:.4f} flow={flow_x:.4f}"
            f" (y {frame_y:.4f} {flow_y:.4f}) grad_norm={grad_norm:.3f}"
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
            loss_names=np.array(["frame", "flow", "grad_norm"]),
            catalog=catalog_history[:done],
            catalog_names=np.array(
                ["frame_x", "flow_x", "frame_y", "flow_y", "w_rms", "z_x"]
            ),
            frame_zoom=frame_history[:done, 0],
            frame_calibration=frame_history[:done, 1],
            f0_fisher=fisher_history[:done],
            flow_zoom=flow_history[:done, 0],
            flow_calibration=flow_history[:done, 1],
            groups=np.array(["f0", "fdot", "sky", "z+", "z-"]),
            snr_edges=np.linspace(*LOSS_LOG10_SNR, LOSS_SNR_BINS + 1),
            time_edges=LOSS_TIME_DECADES
            * (np.arange(LOSS_TIME_BINS + 1) / LOSS_TIME_BINS - 1),
        )
        os.replace(run_dir / "history.tmp.npz", run_dir / "history.npz")

    checkpoints.close()
    print(f"[done] {config.run} -> {run_dir}", flush=True)


if __name__ == "__main__":
    main()
