"""Sanity check: point-estimate regression x_hat = f(y) on the multi-source,
all-8-param problem.

Trains a direct regressor (same MMDiT backbone as train.py) to predict every
source's full unit-cube parameter vector (f0, fdot, A, ra, dec, psi, iota,
phi0) from the WDM conditioning y, isolating "is y digestible" from "does flow
matching converge". Uses lisa.get_train_batch, which returns the full 8-param
geodesic.

Sources are exchangeable (nothing about the WDM conditioning tells the network
which physical source is "slot 0" vs "slot 1"), so both the training loss and
the eval R2/plot are permutation-invariant: every permutation of the source
order is scored and the cheapest assignment is used, per sample. This scales as
N_SOURCES! permutations, so it gets expensive for large N_SOURCES.

Knobs (env): N_SOURCES (>= 1, default 1), NOISE_SCALE (default 1.0, read
inside lisa; set 0.0 for the noiseless case).
Target: x1 = xt + (1 - t) * dx recovers the t=1 unit-cube endpoint.
Periodic checkpoint + eval every CHECKPOINT_EVERY_s so long runs are crash-safe.
"""

import argparse
import functools
import itertools
import os
import time
import warnings

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
import optax
from tqdm import tqdm

jax.config.update("jax_enable_x64", True)

from src import lisa, networks

SEED = 0
N_SOURCES = int(os.environ.get("N_SOURCES", "1"))
assert N_SOURCES >= 1, "N_SOURCES must be >= 1"
if N_SOURCES > 4:
    warnings.warn(
        f"N_SOURCES={N_SOURCES}: the permutation-invariant loss scores all "
        f"{N_SOURCES}! permutations of the source order, which scales "
        "factorially and will be very slow and memory-hungry.",
        stacklevel=2,
    )
T_OBS = lisa.MONTH_s
ALL_LABELS = ["f0", "fdot", "A", "ra", "dec", "psi", "iota", "phi0"]

HIDDEN_DIM = 256
NUM_BLOCKS = 4
NUM_HEADS = 8

LEARNING_RATE = 3e-4
BATCH_SIZE = 256
NOISE_SCALE = 0.0
TAG = f"{N_SOURCES}src_all_noise{NOISE_SCALE:g}"
CHECKPOINT_PATH = f"checkpoint_regression_{TAG}.eqx"
LOSS_PLOT_PATH = f"training_loss_regression_{TAG}.pdf"
RECON_PLOT_PATH = f"regression_reconstruction_{TAG}.pdf"

x_dim = 8
PARAM_NAMES = [f"{p}_{s}" for s in range(N_SOURCES) for p in ALL_LABELS]

# all permutations of the source axis, for the exchangeable-source loss/eval
PERMS = np.array(list(itertools.permutations(range(N_SOURCES))))  # (P, S)
PERMS_J = jnp.asarray(PERMS)


def main(seconds: float):
    print(f"JAX backend: {jax.default_backend()}, devices: {jax.local_device_count()}")
    print(f"N_SOURCES={N_SOURCES}  NOISE_SCALE={NOISE_SCALE:g}")
    key = jr.key(SEED)

    key, key_mock = jr.split(key)
    xt, dx, t, y = lisa.get_train_batch(
        key_mock, batch_size=2, n_sources=N_SOURCES, t_obs=T_OBS
    )
    print(f"x_dim={x_dim}  y_dim={y.shape[-1]}  n_sources={N_SOURCES}  y={y.shape[1:]}")

    key, key_init = jr.split(key)
    net = networks.MMDiT(
        x_dim=x_dim,
        y_dim=y.shape[-1],
        hidden_dim=HIDDEN_DIM,
        num_blocks=NUM_BLOCKS,
        num_heads=NUM_HEADS,
        key=key_init,
    )
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(LEARNING_RATE))
    opt_state = optimizer.init(eqx.filter(net, eqx.is_array))

    query = jnp.zeros((N_SOURCES, x_dim))
    t_fixed = jnp.array(0.0)

    @functools.partial(eqx.filter_jit, donate="all")
    def train_step(net, opt_state, y, x1):
        def loss_fn(net):
            pred = jax.vmap(lambda yi: net(query, yi, t_fixed))(y)  # (B, S, x_dim)
            # sources are exchangeable: score every permutation of the source
            # order and take whichever assignment is cheapest, per sample
            pred_perm = pred[:, PERMS_J, :]  # (B, P, S, x_dim)
            se = jnp.mean((pred_perm - x1[:, None]) ** 2, axis=(-2, -1))  # (B, P)
            return jnp.mean(jnp.min(se, axis=1))

        loss, grads = eqx.filter_value_and_grad(loss_fn)(net)
        updates, opt_state = optimizer.update(
            grads, opt_state, eqx.filter(net, eqx.is_array)
        )
        net = eqx.apply_updates(net, updates)
        return net, opt_state, loss

    @eqx.filter_jit
    def predict(net, y):
        return jnp.clip(jax.vmap(lambda yi: net(query, yi, t_fixed))(y), 0.0, 1.0)

    # fixed held-out eval batch (same each checkpoint so numbers are comparable)
    key, key_eval = jr.split(key)
    xe, dxe, te, ye = lisa.get_train_batch(
        key_eval,
        batch_size=512,
        n_sources=N_SOURCES,
        t_obs=T_OBS,
        noise_scale=NOISE_SCALE,
    )
    x1_eval = np.asarray(xe + (1.0 - te)[:, None, None] * dxe)

    def checkpoint_and_eval(net, losses, elapsed):
        eqx.tree_serialise_leaves(CHECKPOINT_PATH, net)

        plt.figure()
        plt.loglog(losses)
        plt.xlabel("step")
        plt.ylabel("MSE loss")
        plt.grid()
        plt.title(f"x|y regression ({TAG})")
        plt.savefig(LOSS_PLOT_PATH)
        plt.close()

        pred = np.asarray(predict(net, ye))
        # align each sample's predicted source order to whichever permutation
        # best matches the truth
        pred_perm = pred[:, PERMS, :]  # (B, P, S, x_dim)
        se = np.mean((pred_perm - x1_eval[:, None]) ** 2, axis=(2, 3))  # (B, P)
        best = np.argmin(se, axis=1)  # (B,)
        pred = pred_perm[np.arange(pred.shape[0]), best]  # (B, S, x_dim)

        # flatten (B, S, x_dim) -> (B, S*x_dim), matching PARAM_NAMES' source-major order
        true_flat = x1_eval.reshape(x1_eval.shape[0], -1)
        pred_flat = pred.reshape(pred.shape[0], -1)
        r2s = []
        for i, name in enumerate(PARAM_NAMES):
            tr, pr = true_flat[:, i], pred_flat[:, i]
            r2 = 1 - np.sum((pr - tr) ** 2) / np.sum((tr - tr.mean()) ** 2)
            r2s.append(f"{name} R2={r2:.3f} (|err|={np.median(np.abs(pr-tr)):.3f})")
        print(
            f"[{elapsed/3600:.2f} h] loss={np.mean(losses[-200:]):.5f}  "
            + "  ".join(r2s),
            flush=True,
        )

        fig, axes = plt.subplots(
            N_SOURCES, x_dim, figsize=(3 * x_dim, 3 * N_SOURCES), squeeze=False
        )
        for i, (ax, name) in enumerate(zip(axes.flat, PARAM_NAMES)):
            ax.scatter(true_flat[:, i], pred_flat[:, i], s=3, alpha=0.3)
            ax.plot([0, 1], [0, 1], "k--", lw=1)
            ax.set_xlabel(f"true {name}")
            ax.set_ylabel(f"pred {name}")
            ax.set_title(name)
        fig.suptitle(f"x|y reconstruction ({TAG}, {elapsed/3600:.1f}h)")
        fig.tight_layout()
        fig.savefig(RECON_PLOT_PATH)
        plt.close(fig)

    CHECKPOINT_EVERY_s = float(os.environ.get("CHECKPOINT_EVERY_s", "300"))
    losses = []
    t0 = time.monotonic()
    last_save = t0
    pbar = tqdm(total=seconds, unit="s", desc="training")
    while time.monotonic() - t0 < seconds:
        key, key_batch = jr.split(key)
        xt, dx, t, y = lisa.get_train_batch(
            key_batch,
            batch_size=BATCH_SIZE,
            n_sources=N_SOURCES,
            t_obs=T_OBS,
            noise_scale=NOISE_SCALE,
        )
        x1 = xt + (1.0 - t)[:, None, None] * dx
        net, opt_state, loss = train_step(net, opt_state, y, x1)
        losses.append(loss.item())
        pbar.update(int(time.monotonic() - t0 - pbar.n))
        pbar.set_postfix(loss=f"{loss.item():.5f}")
        if time.monotonic() - last_save >= CHECKPOINT_EVERY_s:
            checkpoint_and_eval(net, losses, time.monotonic() - t0)
            last_save = time.monotonic()
    pbar.close()

    checkpoint_and_eval(net, losses, time.monotonic() - t0)
    print(f"[checkpoint] final saved -> {CHECKPOINT_PATH}")
    print(f"saved {RECON_PLOT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=1800.0)
    args = parser.parse_args()
    main(args.seconds)
