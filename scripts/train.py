"""Train the flow-matching posterior p(x | y) and checkpoint."""

import functools
import os
import time
from typing import Callable, Optional

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
import optax
from flax import nnx
from flax import serialization
from tqdm import tqdm

jax.config.update("jax_enable_x64", True)

from canna import networks
from canna import lisa
from canna import flow_utils


def env(name: str, default):
    """Read config value ``NAME`` from the environment, cast to the default's type."""
    raw = os.environ.get(name.upper())
    return default if raw is None else type(default)(raw)


# per-run knobs (overridden from the environment, e.g. via sbatch --export)
SEED = env("seed", 0)
NETWORK_DTYPE = env("network_dtype", "float32")  # net precision (data gen stays fp64)
RUN_SECONDS = env("run_seconds", 86400.0)  # 24h
CHECKPOINT_INTERVAL = env("checkpoint_interval", 60.0)
TAG_SUFFIX = env("tag_suffix", "")
OUTPUT_DIR = env("output_dir", "outputs")  # figures, tables, loss curves
CKPT_DIR = env("ckpt_dir", "checkpoints")  # *.msgpack model checkpoints only

# fixed hyperparameters (n_sources, t_obs live in lisa, set by SIMPLIFIED_PROBLEM)
HIDDEN_DIM = 512
NUM_BLOCKS = 4
NUM_HEADS = 8
BATCH_SIZE = env("batch_size", 256)
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 1.0
WARMUP_FRAC = 0.5  # flow-vs-aux weight ramps 0->1 over WARMUP_FRAC of the budget
VAR_EMA_DECAY = 0.99  # smooths the per-target variance normalizers


def loss_weight_schedule(elapsed: float) -> float:
    """Raised-cosine ramp: 0 at elapsed=0 up to 1 at WARMUP_FRAC*RUN_SECONDS, then holds."""
    warmup = max(WARMUP_FRAC * RUN_SECONDS, 1e-8)
    frac = min(max(elapsed / warmup, 0.0), 1.0)
    return 1.0 - 0.5 * (1.0 + np.cos(np.pi * frac))


def make_tag() -> str:
    problem = "simplified" if lisa.SIMPLIFIED_PROBLEM else "full"
    return f"{lisa.N_SOURCES}src_{problem}{TAG_SUFFIX}"


@functools.partial(jax.jit, static_argnums=(1, 2))
def sample_physics_batch(key, batch_size: int, get_physics_sample: Callable):
    """Vmap the injected physics simulator to draw a batch of targets + WDM conditioning."""
    return jax.vmap(get_physics_sample)(jr.split(key, batch_size))


@nnx.jit(
    static_argnames=("ema_decay", "net_dtype", "geodesic", "match_sources"),
    donate_argnames=("flow", "optimizer", "var_ema"),
)
def train_step(
    flow,
    optimizer,
    var_ema,
    key,
    u_targ,
    y,
    y_targ,
    loss_weight,
    ema_decay: float,
    net_dtype,
    geodesic: Optional[Callable] = None,
    match_sources: Optional[Callable] = None,
):
    """One EMA-variance-reweighted optimizer step (flow and optimizer are updated in place)."""
    sample_flow = lambda k, u1: flow_utils.flow_train_sample(
        k, u1, geodesic, match_sources
    )
    ut, du, t = jax.vmap(sample_flow)(jr.split(key, u_targ.shape[0]), u_targ)
    # the network runs in net_dtype (y is pre-cast at batch fetch); loss targets
    # (du, y_targ, u_targ) stay at LISA precision
    ut, t = ut.astype(net_dtype), t.astype(net_dtype)

    @nnx.vmap(in_axes=(None, 0, 0, 0))
    def batched_flow(flow, ut, y, t):
        return flow(ut, y, t)

    def objective(flow):
        du_pred, u1_pred, y_recon = batched_flow(flow, ut, y, t)
        l_flow, v_flow = flow_utils.loss_flow_matching(du_pred, du)
        l_reg_y, v_reg_y = flow_utils.loss_signal_regression(y_recon, y_targ)
        l_reg_u, v_reg_u = flow_utils.loss_param_regression(
            u1_pred, u_targ, match_sources
        )
        target_var = jnp.stack([v_flow, v_reg_u, v_reg_y])
        # rescale each term by its running EMA variance, then weight flow vs auxiliaries
        sub_losses = jnp.stack([l_flow, l_reg_u, l_reg_y]) / (var_ema + 1e-8)
        weights = jnp.array([loss_weight, 1.0 - loss_weight, 1.0 - loss_weight])
        return jnp.sum(sub_losses * weights), (sub_losses, target_var)

    (loss, (sub_losses, target_var)), grads = nnx.value_and_grad(
        objective, has_aux=True
    )(flow)
    optimizer.update(flow, grads)
    var_ema = ema_decay * var_ema + (1.0 - ema_decay) * target_var
    return var_ema, loss, sub_losses


LOSS_COLORS = {"total": "#3d405b", "reg_u": "#81b29a", "reg_y": "#e0a458", "flow": "#e07a5f"}


def ema(x: np.ndarray, span: int = 200) -> np.ndarray:
    """Exponential moving average; span sets the smoothing window (in steps)."""
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return x
    alpha = 2.0 / (span + 1.0)
    out = np.empty_like(x)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = alpha * x[i] + (1.0 - alpha) * out[i - 1]
    return out


if __name__ == "__main__":
    assert 1 <= lisa.N_SOURCES <= 4, "n_sources must be in [1, 4]"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)
    tag = make_tag()
    checkpoint_path = os.path.join(CKPT_DIR, f"checkpoint_flow_{tag}.msgpack")
    loss_plot_path = os.path.join(OUTPUT_DIR, f"training_loss_flow_{tag}.pdf")

    print(f"JAX backend: {jax.default_backend()}, devices: {jax.local_device_count()}")
    print(f"inferring {lisa.PARAMETER_NAMES}")
    key = jr.key(SEED)

    key, key_mock = jr.split(key)
    u1, params, datastream, signal, y, y_clean = sample_physics_batch(
        key_mock, 2, lisa.get_physics_sample
    )
    print(f"{lisa.N_SOURCES} sources, shapes:")
    for name, el in dict(u1=u1, y=y, y_clean=y_clean).items():
        print(f"  {name}: {el.shape} {el.dtype} [{el.min():.3g}, {el.max():.3g}]")

    # network runs in NETWORK_DTYPE (data generation stays fp64 for signal/WDM accuracy)
    net_dtype = jnp.dtype(NETWORK_DTYPE)
    key, key_init = jr.split(key)
    flow = networks.MMDiT(
        x_dim=u1.shape[-1],
        y_channels=y.shape[-1],
        hidden_dim=HIDDEN_DIM,
        num_blocks=NUM_BLOCKS,
        num_heads=NUM_HEADS,
        rngs=nnx.Rngs(key_init),
        dtype=net_dtype,
    )
    optimizer = nnx.Optimizer(
        flow,
        optax.chain(
            optax.clip_by_global_norm(GRAD_CLIP),
            optax.adamw(LEARNING_RATE, weight_decay=WEIGHT_DECAY),
        ),
        wrt=nnx.Param,
    )
    var_ema = jnp.ones(3)

    # loss history: scalars stay on-device in *_pending (no per-step host sync),
    # drained to the host lists in bulk at each checkpoint
    losses: list[float] = []
    sub_losses_log: list[list[float]] = []
    losses_pending: list = []
    sub_losses_pending: list = []

    def drain_losses():
        if not losses_pending:
            return
        losses.extend(np.asarray(jax.device_get(losses_pending)).tolist())
        sub_losses_log.extend(
            np.asarray(jax.device_get(sub_losses_pending)).reshape(-1, 3).tolist()
        )
        losses_pending.clear()
        sub_losses_pending.clear()

    def checkpoint(elapsed: float):
        with open(checkpoint_path, "wb") as f:
            f.write(serialization.to_bytes(nnx.to_pure_dict(nnx.state(flow))))
        drain_losses()

        sub = np.asarray(sub_losses_log)  # (steps, 3): flow, reg_u, reg_y (var-normalized)
        span = max(20, len(losses) // 100)
        steps = np.arange(1, len(losses) + 1)

        fig, ax = plt.subplots()
        # auxiliary losses: smoothed and transparent, so the flow loss reads clearly on top
        for name, y in (("total", np.asarray(losses)), ("reg_u", sub[:, 1]), ("reg_y", sub[:, 2])):
            ax.loglog(steps, ema(y, span), color=LOSS_COLORS[name], alpha=0.5, lw=1.4, label=name)
        # flow loss drawn last, opaque, over a faint raw trace
        ax.loglog(steps, sub[:, 0], color=LOSS_COLORS["flow"], alpha=0.12, lw=0.8)
        ax.loglog(steps, ema(sub[:, 0], span), color=LOSS_COLORS["flow"], lw=2.2, label="flow", zorder=5)
        ax.set_xlabel("step")
        ax.set_ylabel("loss / var_ema")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(frameon=False)
        ax.set_title(f"flow p(x|y) ({tag})")
        fig.savefig(loss_plot_path, bbox_inches="tight")
        plt.close(fig)

        flow_l, reg_u_l, reg_y_l = sub[-200:].mean(axis=0)
        print(
            f"[{elapsed/3600:.2f} h] loss={np.mean(losses[-200:]):.5f} "
            f"(flow={flow_l:.5f} reg_u={reg_u_l:.5f} reg_y={reg_y_l:.5f}) "
            f"w={loss_weight_schedule(elapsed):.3f} var_ema={np.asarray(var_ema)}",
            flush=True,
        )

    LOG_INTERVAL = 20  # steps between live-loss refreshes (each forces one host sync)
    t0 = time.monotonic()
    last_save = t0
    step = 0
    pbar = tqdm(total=RUN_SECONDS, unit="s", desc="training")
    while (elapsed := time.monotonic() - t0) < RUN_SECONDS:
        key, key_phys, key_step = jr.split(key, 3)
        u1, params, datastream, signal, y, y_clean = sample_physics_batch(
            key_phys, BATCH_SIZE, lisa.get_physics_sample
        )
        # cast the network conditioning down to train precision at the boundary
        # (avoids feeding the fp64 batch into the donated step; targets stay fp64)
        y = y.astype(net_dtype)
        loss_weight = jnp.float32(loss_weight_schedule(elapsed))
        var_ema, loss, sub_losses = train_step(
            flow,
            optimizer,
            var_ema,
            key_step,
            u1,
            y,
            y_clean,
            loss_weight,
            VAR_EMA_DECAY,
            net_dtype,
            lisa.geodesic,
            lisa.match_sources,
        )
        # buffer the scalars on-device; no .item() here (a per-step host sync
        # would serialize next-batch generation behind the current step)
        losses_pending.append(loss)
        sub_losses_pending.append(sub_losses)
        step += 1

        pbar.update(int(time.monotonic() - t0 - pbar.n))
        # refresh the live loss only occasionally -- each refresh forces one sync
        if step % LOG_INTERVAL == 0:
            loss_flow, loss_reg_x, loss_reg_y = (float(s) for s in sub_losses)
            pbar.set_postfix(
                loss=f"{float(loss):.5f}",
                flow=f"{loss_flow:.5f}",
                reg_x=f"{loss_reg_x:.5f}",
                reg_y=f"{loss_reg_y:.5f}",
                w=f"{loss_weight:.2f}",
            )
        if time.monotonic() - last_save >= CHECKPOINT_INTERVAL:
            checkpoint(time.monotonic() - t0)
            last_save = time.monotonic()
    pbar.close()

    checkpoint(time.monotonic() - t0)
    print(f"[checkpoint] final saved -> {checkpoint_path}")
