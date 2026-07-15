"""Train the flow-matching posterior p(x | y) for LISA Galactic Binaries and checkpoint.

Hard-coded for LISA; the only env knobs are the import-time toggles in canna.lisa
(SIMPLIFIED_PROBLEM, N_SOURCES). Everything else lives in the config block below.
"""

from collections.abc import Callable
from typing import NamedTuple
from jaxtyping import Array, Float, Scalar
import os
import signal

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
import optax
from flax import nnx, serialization

from tqdm import tqdm

jax.config.update("jax_enable_x64", True)

from canna import lisa, networks

SEED = 0
TAG_SUFFIX = ""
OUTPUT_DIR = "outputs"
CHECKPOINT_DIR = "checkpoints"
LOG_INTERVAL = 100
CHECKPOINT_INTERVAL = 1000

HIDDEN_DIM = 512
NUM_BLOCKS = 8
NUM_HEADS = 8

BATCH_SIZE = 256
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
NETWORK_PARAM_DTYPE = jnp.float32
NETWORK_COMPUTE_DTYPE = jnp.float32  # TODO: mixedprecision training

TOTAL_TRAIN_STEPS = 500_000
WARMUP_FRAC = 0.5  # auxiliary loss weight decays 1->0 over this fraction of the run

LOSS_COLORS = {"flow": "#2a78d6", "reg_u": "#199e70", "reg_y": "#4a3aa7"}


def install_walltime_stop() -> Callable[[], bool]:
    """Arm SIGUSR1 -- slurm sends it before the walltime -- as a request to stop; returns a predicate."""
    stop_requested = False

    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True
        print("\n[signal] SIGUSR1: stopping at the next log point", flush=True)

    signal.signal(signal.SIGUSR1, request_stop)
    return lambda: stop_requested


class TrainBatch(NamedTuple):
    ut: Float[Array, "B S P"]
    du: Float[Array, "B S P"]
    t: Float[Array, "B"]
    y: Float[Array, "B T F C"]
    u1: Float[Array, "B S P"]
    y_clean: Float[Array, "B T F C"]


# insertion order fixes the variance() stacking order
TARGET_NAMES = ("flow", "reg_u", "reg_y")


class TargetStats(nnx.MultiMetric):
    """Running variance of each loss target, accumulated over batches with Welford's algorithm."""

    def __init__(self):
        # each Welford reads its own argname out of the broadcast update kwargs
        super().__init__(
            **{name: nnx.metrics.Welford(argname=name) for name in TARGET_NAMES}
        )

    def update(self, batch: TrainBatch):
        """Fold each target's per-batch variance into its running estimate."""
        targets = (batch.du, batch.u1, batch.y_clean)
        super().update(
            **{
                name: jnp.var(target).astype(jnp.float32)
                for name, target in zip(TARGET_NAMES, targets)
            }
        )

    def variance(self) -> Float[Array, "3"]:
        """Target variance per loss term, in TARGET_NAMES order.

        Each Welford is fed one variance per batch, so its running *mean* is the variance estimate.
        """
        return jnp.stack([stat.mean for stat in self.compute().values()])


def setup(ckpt_path: str) -> tuple[networks.MMDiT, nnx.Optimizer, TargetStats, int]:
    """Build the flow, optimizer, and target stats, resuming from `ckpt_path` if it exists.

    Returns the step to resume from: 0 unless a checkpoint was restored.
    """
    flow = networks.MMDiT(
        x_dim=len(lisa.PARAMETER_NAMES),
        y_channels=len(lisa.CHANNEL_NAMES),
        hidden_dim=HIDDEN_DIM,
        num_blocks=NUM_BLOCKS,
        num_heads=NUM_HEADS,
        dtype=NETWORK_COMPUTE_DTYPE,
        param_dtype=NETWORK_PARAM_DTYPE,
        rngs=nnx.Rngs(SEED),
    )
    optimizer = nnx.Optimizer(
        model=flow,
        tx=optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(LEARNING_RATE, weight_decay=WEIGHT_DECAY),
        ),
        wrt=nnx.Param,
    )
    stats = TargetStats()
    if not os.path.exists(ckpt_path):
        return flow, optimizer, stats, 0
    return flow, optimizer, stats, restore_checkpoint(flow, optimizer, stats, ckpt_path)


@nnx.jit(static_argnames=("batch_size",))
def sample_train_batch(rngs: nnx.Rngs, batch_size: int) -> TrainBatch:
    """Sample a training batch."""

    @nnx.split_rngs(splits=batch_size)
    @nnx.vmap(axis_size=batch_size)
    def sample_single(rngs: nnx.Rngs):
        u1, _, _, _, y, y_clean = lisa.get_physics_sample(rngs())

        # couple a base point to the target
        u0 = rngs.uniform(u1.shape, u1.dtype)
        u0, _ = lisa.match_sources(u0, u1)

        # sample the conditional probability path
        t = rngs.uniform(dtype=u1.dtype)
        ut = lisa.geodesic(t, u0, u1)
        du = jax.jacobian(lisa.geodesic)(t, u0, u1)
        batch = ut, du, t, y, u1, y_clean

        # cast to the network compute dtype
        batch = tuple(el.astype(NETWORK_COMPUTE_DTYPE) for el in batch)
        return batch

    return TrainBatch(*sample_single(rngs))


@nnx.jit(donate_argnames=("flow", "optimizer"))
def train_step(
    flow: networks.MMDiT,
    optimizer: nnx.Optimizer,
    stats: TargetStats,
    batch: TrainBatch,
    aux_weight: float,
) -> Float[Array, "3"]:
    """One optimizer step on `batch`; mutates flow/optimizer/stats in place. Returns raw per-term losses."""
    # fold in this batch before reading, so step 0 already has a variance estimate
    stats.update(batch)
    target_var = stats.variance()

    def train_loss(flow: networks.MMDiT) -> tuple[Scalar, Float[Array, "3"]]:
        du_pred, u_pred, y_recon = flow(batch.ut, batch.y, batch.t)
        u_pred, _ = jax.vmap(lisa.match_sources)(u_pred, batch.u1)
        assert all(a == c for a, c in zip(y_recon.shape, batch.y_clean.shape))
        l_flow = jnp.mean(optax.l2_loss(du_pred, batch.du))
        l_reg_u = jnp.mean(optax.l2_loss(u_pred, batch.u1))
        l_reg_y = jnp.mean(optax.l2_loss(y_recon, batch.y_clean))
        losses = jnp.stack([l_flow, l_reg_u, l_reg_y])

        # normalize by target variance so the three terms are comparable before weighting
        weights = jnp.array([1.0, aux_weight, aux_weight])
        return jnp.sum(weights * losses / (target_var + 1e-8)), losses

    (_, losses), grads = nnx.value_and_grad(train_loss, has_aux=True)(flow)
    optimizer.update(flow, grads)
    return losses


def aux_loss_weight_schedule(step: int) -> float:
    """Compute auxiliary loss weight, decaying from 1->0 over WARMUP_FRAC of the run."""
    frac = min(step / (WARMUP_FRAC * TOTAL_TRAIN_STEPS), 1.0)
    return 0.5 + 0.5 * np.cos(np.pi * frac)


def save_checkpoint(
    flow: networks.MMDiT,
    optimizer: nnx.Optimizer,
    stats: TargetStats,
    ckpt_path: str,
):
    """Save flow weights plus optimizer state (which carries the step count) and target stats."""
    ckpt = {
        "flow": nnx.to_pure_dict(nnx.state(flow)),
        "opt": nnx.to_pure_dict(nnx.state(optimizer)),
        "stats": nnx.to_pure_dict(nnx.state(stats)),
    }
    # rename onto the path so a concurrent reader never sees a half-written file
    tmp_path = f"{ckpt_path}.tmp"
    with open(tmp_path, "wb") as f:
        f.write(serialization.to_bytes(ckpt))
    os.replace(tmp_path, ckpt_path)


def restore_checkpoint(
    flow: networks.MMDiT,
    optimizer: nnx.Optimizer,
    stats: TargetStats,
    ckpt_path: str,
) -> int:
    """Restore flow, optimizer, and target stats in place from `ckpt_path`; returns the step to resume from."""
    with open(ckpt_path, "rb") as f:
        ckpt = serialization.msgpack_restore(f.read())
    if not {"flow", "opt", "stats"} <= set(ckpt):
        raise ValueError(
            f"{ckpt_path} predates the current checkpoint format and has no optimizer or "
            f"target-stats state to resume from; move it aside to train from scratch"
        )

    for module, pure_dict in (
        (flow, ckpt["flow"]),
        (optimizer, ckpt["opt"]),
        (stats, ckpt["stats"]),
    ):
        state = nnx.state(module)
        nnx.replace_by_pure_dict(state, pure_dict)
        nnx.update(module, state)
    return int(optimizer.step[...])


def save_loss_history(loss_hist: list[tuple[float, float, float]], path: str):
    """Persist the loss curve so a requeued attempt plots the whole run and not just its own slice."""
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "wb") as f:
        np.save(f, np.asarray(loss_hist))
    os.replace(tmp_path, path)


def load_loss_history(path: str, start_step: int) -> list[tuple[float, float, float]]:
    """Loss points from earlier attempts, trimmed to the step the checkpoint resumed at."""
    if start_step == 0 or not os.path.exists(path):
        return []
    with open(path, "rb") as f:
        hist = np.load(f)
    return [tuple(point) for point in hist[: start_step // LOG_INTERVAL]]


def plot_loss_curve(
    loss_hist: list[tuple[float, float, float]], tag: str, loss_path: str
):
    hist = np.asarray(loss_hist)
    xs = (np.arange(len(hist)) + 1) * LOG_INTERVAL
    fig, ax = plt.subplots()
    ax.loglog(xs, hist[:, 0], color=LOSS_COLORS["flow"], lw=2, label="flow")
    ax.loglog(xs, hist[:, 1], color=LOSS_COLORS["reg_u"], label="reg_u", alpha=0.3)
    ax.loglog(xs, hist[:, 2], color=LOSS_COLORS["reg_y"], label="reg_y", alpha=0.3)
    ax.set(xlabel="step", ylabel="loss", title=f"flow p(x|y) ({tag})")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(loss_path, bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    tag = f"{lisa.N_SOURCES}src_{"simplified" if lisa.SIMPLIFIED_PROBLEM else "full"}{TAG_SUFFIX}"
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_flow_{tag}.msgpack")
    loss_path = os.path.join(OUTPUT_DIR, f"training_loss_flow_{tag}.pdf")
    hist_path = os.path.join(OUTPUT_DIR, f"loss_history_flow_{tag}.npy")
    walltime_reached = install_walltime_stop()

    print(
        f"JAX backend: {jax.default_backend()}, devices: {jax.local_device_count()}",
        flush=True,
    )
    print(f"inferring {lisa.PARAMETER_NAMES}", flush=True)

    flow, optimizer, stats, start_step = setup(ckpt_path)
    # fold the resume point into the seed, or a requeued run replays the data it already saw
    rngs = nnx.Rngs(jr.fold_in(jr.key(SEED), start_step))
    if start_step:
        print(f"[checkpoint] resuming {ckpt_path} at step {start_step}", flush=True)

    # --- sample a batch and print its shapes, dtypes, and ranges ---
    batch = sample_train_batch(rngs, BATCH_SIZE)
    for name, x in batch._asdict().items():
        print(
            f"  {name}: {x.shape} {x.dtype} [{float(x.min()):.3g}, {float(x.max()):.3g}]",
            flush=True,
        )

    # --- training loop ---
    # per-interval mean [flow, reg_u, reg_y] losses (host), carried across requeues
    loss_hist = load_loss_history(hist_path, start_step)
    steps = range(start_step, TOTAL_TRAIN_STEPS, LOG_INTERVAL)
    for step in (
        pbar := tqdm(
            steps,
            initial=start_step // LOG_INTERVAL,
            total=TOTAL_TRAIN_STEPS // LOG_INTERVAL,
        )
    ):
        buffer = []  # per-step losses since the last log point
        aux_weight = aux_loss_weight_schedule(step)

        for _ in range(LOG_INTERVAL):
            batch = sample_train_batch(rngs, BATCH_SIZE)
            losses = train_step(flow, optimizer, stats, batch, aux_weight)
            buffer.append(losses)

        # ---- log the interval-mean loss point ----
        buffer = jnp.mean(jnp.stack(buffer), axis=0)
        flow_l, reg_u_l, reg_y_l = jax.device_get(buffer)
        loss_hist.append((flow_l, reg_u_l, reg_y_l))
        pbar.set_postfix(
            flow=f"{flow_l:.5f}", reg_u=f"{reg_u_l:.5f}", reg_y=f"{reg_y_l:.5f}"
        )

        # ---- checkpoint + loss curve every CHECKPOINT_INTERVAL steps ----
        if (step + LOG_INTERVAL) % CHECKPOINT_INTERVAL == 0:
            save_checkpoint(flow, optimizer, stats, ckpt_path)
            save_loss_history(loss_hist, hist_path)
            plot_loss_curve(loss_hist, tag, loss_path)
            print(
                f"[step {step + LOG_INTERVAL}/{TOTAL_TRAIN_STEPS}] flow={flow_l:.5f} reg_u={reg_u_l:.5f} reg_y={reg_y_l:.5f}",
                flush=True,
            )

        # train.slurm requeues the job when we stop here; the save below is what it resumes from
        if walltime_reached():
            break

    save_checkpoint(flow, optimizer, stats, ckpt_path)
    save_loss_history(loss_hist, hist_path)
    plot_loss_curve(loss_hist, tag, loss_path)
    print(
        f"[checkpoint] saved at step {int(optimizer.step[...])} -> {ckpt_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
