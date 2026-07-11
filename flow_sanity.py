"""Flow-matching sanity check: posterior p(x | y), multi-source.

SIMPLIFIED=1 restricts inference to [f0, fdot, A, psi], marginalizing the rest as
nuisances (via the lisa_simplified drop-in); otherwise the full 8-param problem.
The two share this script: a single flag swaps the lisa interface and output tag.

Knobs (env): N_SOURCES (>= 1, default 1), MIN_SNR (start, default 100.0),
MIN_SNR_FINAL (end, default 1.0), RUN_SECONDS, CHECKPOINT_INTERVAL,
SIMPLIFIED, TAG_SUFFIX, SNR_WARMUP_FRAC.

Difficulty is scheduled via a minimum-SNR floor: weak injections are amplified
up to this floor so early batches are easy (loud, well-detected signals). The
floor ramps (raised-cosine in log-space, so smoothly at both ends) from
MIN_SNR down to MIN_SNR_FINAL once SNR_WARMUP_FRAC * RUN_SECONDS has elapsed,
then holds.

This script only trains and checkpoints; it does not draw posteriors or plot
corner plots (that batch is too big to fit alongside training state on most
GPUs). Run eval_corner.py separately against a saved checkpoint for that.
"""

import functools
import os
import time

import equinox as eqx
import jax
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
import optax
from tqdm import tqdm

jax.config.update("jax_enable_x64", True)

from src import networks
from src import lisa

# problem knobs (env)
N_SOURCES = int(os.environ.get("N_SOURCES", "1"))
MIN_SNR = float(os.environ.get("MIN_SNR", "100.0"))
MIN_SNR_FINAL = float(os.environ.get("MIN_SNR_FINAL", "1.0"))
SNR_WARMUP_FRAC = float(os.environ.get("SNR_WARMUP_FRAC", "1.0"))
RUN_SECONDS = float(os.environ.get("RUN_SECONDS", "86400"))
T_OBS = lisa.MONTH
CHECKPOINT_INTERVAL = float(os.environ.get("CHECKPOINT_INTERVAL", "1800"))  # [s]
SIMPLIFIED = os.environ.get("SIMPLIFIED", "0") not in ("0", "", "false", "False")

# the SIMPLIFIED flag swaps in a drop-in lisa with x restricted to [f0, fdot, A, psi]
if SIMPLIFIED:
    from src import lisa_simplified as lisa

assert N_SOURCES >= 1, "N_SOURCES must be >= 1"
assert N_SOURCES <= 4, "N_SOURCES must be <= 4"

# flow and training hyperparameters
SEED = 0
HIDDEN_DIM = 256
NUM_BLOCKS = 4
NUM_HEADS = 8
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 512

# checkpoint pathnames
TAG = (
    f"{N_SOURCES}src"
    + ("_simplified" if SIMPLIFIED else "")
    + os.environ.get("TAG_SUFFIX", "")
)
CHECKPOINT_PATH = f"checkpoint_flow_{TAG}.eqx"
LOSS_PLOT_PATH = f"training_loss_flow_{TAG}.pdf"


def minimum_snr_schedule(elapsed):
    """Log-space raised-cosine ramp: MIN_SNR at elapsed=0 down to MIN_SNR_FINAL at
    SNR_WARMUP_FRAC*RUN_SECONDS, then holds."""
    warmup_seconds = SNR_WARMUP_FRAC * RUN_SECONDS
    if warmup_seconds <= 0:
        return MIN_SNR_FINAL
    frac = min(max(elapsed / warmup_seconds, 0.0), 1.0)
    cos_frac = 0.5 * (1.0 + np.cos(np.pi * frac))
    log_snr = np.log(MIN_SNR_FINAL) + (np.log(MIN_SNR) - np.log(MIN_SNR_FINAL)) * cos_frac
    return np.exp(log_snr)


if __name__ == "__main__":
    print(f"JAX backend: {jax.default_backend()}, devices: {jax.local_device_count()}")
    print(
        f"N_SOURCES={N_SOURCES}  MIN_SNR={MIN_SNR:g}->{MIN_SNR_FINAL:g} (log-space, "
        f"ramped over {SNR_WARMUP_FRAC:g} of RUN_SECONDS)  SIMPLIFIED={SIMPLIFIED}"
    )
    print(f"inferring {lisa.PARAMETER_NAMES}")
    key = jr.key(SEED)

    key, key_mock = jr.split(key)
    xt, dx, t, y, x0, x1, params, datastream = lisa.get_train_batch(
        key_mock, batch_size=2, n_sources=N_SOURCES, t_obs=T_OBS
    )
    print(f"x:{xt.shape[1:]}  y:{y.shape[1:]}")

    key, key_init = jr.split(key)
    flow = networks.MMDiT(
        x_dim=xt.shape[-1],
        y_dim=y.shape[-1],
        hidden_dim=HIDDEN_DIM,
        num_blocks=NUM_BLOCKS,
        num_heads=NUM_HEADS,
        key=key_init,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(LEARNING_RATE, weight_decay=WEIGHT_DECAY),
    )
    opt_state = optimizer.init(eqx.filter(flow, eqx.is_array))

    @functools.partial(eqx.filter_jit, donate="all")
    def train_step(flow, opt_state, xt, dx, t, y, x1):
        def loss_fn(flow):
            # compute the flow-matching loss
            dx_pred, x_mle_pred = jax.vmap(flow)(xt, y, t)
            loss_flow = (dx_pred - dx) ** 2
            # auxiliary regression loss on the MLE point estimate of x1
            _, loss_reg = jax.vmap(lisa.match_sources)(x_mle_pred, x1)
            return loss_flow.mean() + loss_reg.mean()

        loss, grads = eqx.filter_value_and_grad(loss_fn)(flow)
        updates, opt_state = optimizer.update(
            grads, opt_state, eqx.filter(flow, eqx.is_array)
        )
        flow = eqx.apply_updates(flow, updates)
        return flow, opt_state, loss

    def checkpoint(flow, losses, elapsed):
        eqx.tree_serialise_leaves(CHECKPOINT_PATH, flow)

        plt.figure()
        plt.loglog(losses)
        plt.xlabel("step")
        plt.ylabel("flow-matching MSE")
        plt.grid()
        plt.title(f"flow p(x|y) ({TAG})")
        plt.savefig(LOSS_PLOT_PATH)
        plt.close()

        print(
            f"[{elapsed/3600:.2f} h] loss={np.mean(losses[-200:]):.5f} "
            f"min_snr={minimum_snr_schedule(elapsed):.3f}",
            flush=True,
        )

    losses = []
    t0 = time.monotonic()
    last_save = t0
    pbar = tqdm(total=RUN_SECONDS, unit="s", desc="training")
    while (elapsed := (time.monotonic() - t0)) < RUN_SECONDS:
        key, key_batch = jr.split(key)
        snr_threshold = minimum_snr_schedule(elapsed)
        xt, dx, t, y, x0, x1, params, datastream = lisa.get_train_batch(
            key_batch,
            batch_size=BATCH_SIZE,
            n_sources=N_SOURCES,
            t_obs=T_OBS,
            snr_threshold=snr_threshold,
        )
        flow, opt_state, loss = train_step(flow, opt_state, xt, dx, t, y, x1)
        losses.append(loss.item())
        pbar.update(int(time.monotonic() - t0 - pbar.n))
        pbar.set_postfix(loss=f"{loss.item():.5f}", min_snr=f"{snr_threshold:.2f}")
        if time.monotonic() - last_save >= CHECKPOINT_INTERVAL:
            checkpoint(flow, losses, time.monotonic() - t0)
            last_save = time.monotonic()
    pbar.close()

    checkpoint(flow, losses, time.monotonic() - t0)
    print(f"[checkpoint] final saved -> {CHECKPOINT_PATH}")
