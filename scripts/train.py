"""Train the flow-matching posterior p(x | y) for LISA Galactic Binaries and checkpoint.

Hard-coded for LISA; the only env knobs are the import-time toggles in canna.lisa
(SIMPLIFIED_PROBLEM, N_SOURCES). Everything else lives in the config block below.
"""

from typing import NamedTuple
from jaxtyping import Array, Float, Scalar
import os

import jax
import jax.numpy as jnp
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


def setup() -> tuple[networks.MMDiT, nnx.Optimizer]:
    """Build the flow network, its LR schedule, and the optimizer."""
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
    return flow, optimizer


class TrainBatch(NamedTuple):
    ut: Float[Array, "B S P"]
    du: Float[Array, "B S P"]
    t: Float[Array, "B"]
    y: Float[Array, "B T F C"]
    u1: Float[Array, "B S P"]
    y_clean: Float[Array, "B T F C"]


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
    batch: TrainBatch,
    aux_weight: float,
) -> Float[Array, "3"]:
    """One optimizer step on `batch`; mutates flow/optimizer in place. Returns raw per-term losses."""

    def train_loss(flow: networks.MMDiT) -> tuple[Scalar, Float[Array, "3"]]:
        du_pred, u_pred, y_recon = flow(batch.ut, batch.y, batch.t)
        u_pred, _ = jax.vmap(lisa.match_sources)(u_pred, batch.u1)
        assert all(a == c for a, c in zip(y_recon.shape, batch.y_clean.shape))
        l_flow = jnp.mean(optax.l2_loss(du_pred, batch.du))
        l_reg_u = jnp.mean(optax.l2_loss(u_pred, batch.u1))
        l_reg_y = jnp.mean(optax.l2_loss(y_recon, batch.y_clean))
        losses = jnp.stack([l_flow, l_reg_u, l_reg_y])
        return l_flow + aux_weight * (l_reg_u + l_reg_y), losses

    (_, losses), grads = nnx.value_and_grad(train_loss, has_aux=True)(flow)
    optimizer.update(flow, grads)
    return losses


def aux_loss_weight_schedule(step: int) -> float:
    """Compute auxiliary loss weight, decaying from 1->0 over WARMUP_FRAC of the run."""
    frac = min(step / (WARMUP_FRAC * TOTAL_TRAIN_STEPS), 1.0)
    return 0.5 + 0.5 * np.cos(np.pi * frac)


def save_checkpoint(flow: networks.MMDiT, ckpt_path: str):
    """Save the flow weights and plot the loss curve so far."""
    with open(ckpt_path, "wb") as f:
        f.write(serialization.to_bytes(nnx.to_pure_dict(nnx.state(flow))))


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

    print(f"JAX backend: {jax.default_backend()}, devices: {jax.local_device_count()}")
    print(f"inferring {lisa.PARAMETER_NAMES}")

    rngs = nnx.Rngs(SEED)
    flow, optimizer = setup()

    # --- sample a batch and print its shapes, dtypes, and ranges ---
    batch = sample_train_batch(rngs, BATCH_SIZE)
    for name, x in batch._asdict().items():
        print(
            f"  {name}: {x.shape} {x.dtype} [{float(x.min()):.3g}, {float(x.max()):.3g}]"
        )

    # --- training loop ---
    loss_hist = []  # per-interval mean [flow, reg_u, reg_y] losses (host)
    for step in (pbar := tqdm(range(0, TOTAL_TRAIN_STEPS, LOG_INTERVAL))):
        buffer = []  # per-step losses since the last log point
        aux_weight = aux_loss_weight_schedule(step)

        for _ in range(LOG_INTERVAL):
            batch = sample_train_batch(rngs, BATCH_SIZE)
            losses = train_step(flow, optimizer, batch, aux_weight)
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
            save_checkpoint(flow, ckpt_path)
            plot_loss_curve(loss_hist, tag, loss_path)
            print(
                f"[step {step + LOG_INTERVAL}/{TOTAL_TRAIN_STEPS}] flow={flow_l:.5f} reg_u={reg_u_l:.5f} reg_y={reg_y_l:.5f}"
            )
    save_checkpoint(flow, ckpt_path)
    plot_loss_curve(loss_hist, tag, loss_path)
    print(f"[checkpoint] final saved -> {ckpt_path}")


if __name__ == "__main__":
    main()
