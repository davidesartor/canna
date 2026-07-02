"""Sanity check: is the WDM conditioning signal `y` digestible?

Trains a direct point-estimate regressor `x_hat = f(y)` using the *same*
MMDiT backbone as train.py/test.py, instead of a flow-matching velocity
field. The x-stream input is a fixed all-zeros query (S source tokens); the
network only has `y` to work with, so if it can't reconstruct the true
(f0, fdot, A) per source, the conditioning signal itself doesn't carry that
information and flow-matching training in train.py wouldn't either -- this
isolates "is y informative" from "does the flow-matching objective converge".

Uses lisa.get_train_batch unmodified: x1 = xt + (1 - t) * dx recovers the
t=1 endpoint (the true unit-cube parameters) from the returned geodesic
point, since dx = x1 - x0 exactly.
"""
import argparse
import functools
import os
import time

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

# problem (mirrors train.py). N_SOURCES is env-overridable: with >1 exchangeable
# source the order-sensitive MSE is confounded by label-switching, so use 1 to
# cleanly test whether y carries the parameter information at all.
SEED = 0
N_SOURCES = int(os.environ.get("N_SOURCES", "2"))
T_OBS = lisa.MONTH_s

# model (same size as train.py's MMDiT, for a fair backbone comparison)
HIDDEN_DIM = 256
NUM_BLOCKS = 4
NUM_HEADS = 8

# training
LEARNING_RATE = 1e-4
BATCH_SIZE = 512
CHECKPOINT_PATH = f"checkpoint_regression_{N_SOURCES}src.eqx"
LOSS_PLOT_PATH = f"training_loss_regression_{N_SOURCES}src.pdf"
RECON_PLOT_PATH = f"regression_reconstruction_{N_SOURCES}src.pdf"
PARAM_NAMES = ["f0", "fdot", "A"]


def main(seconds: float):
    print(f"JAX backend: {jax.default_backend()}, devices: {jax.local_device_count()}")
    key = jr.key(SEED)

    key, key_mock = jr.split(key)
    xt, dx, t, y = lisa.get_train_batch(
        key_mock, batch_size=2, n_sources=N_SOURCES, t_obs=T_OBS
    )
    x_dim = xt.shape[-1]
    print(f"x_dim={x_dim}  y_dim={y.shape[-1]}  n_sources={N_SOURCES}")

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

    # Fixed "empty" query tokens: the backbone must recover x purely from y
    # (position embedding still tells the two source tokens apart).
    query = jnp.zeros((N_SOURCES, x_dim))
    # Time is unused for a static regressor but the backbone's sinusoidal
    # embedding expects an array scalar; pin it to 0.
    t_fixed = jnp.array(0.0)

    @functools.partial(eqx.filter_jit, donate="all")
    def train_step(net, opt_state, y, x1):
        def loss_fn(net):
            pred = jax.vmap(lambda yi: net(query, yi, t_fixed))(y)
            return jnp.mean((pred - x1) ** 2)

        loss, grads = eqx.filter_value_and_grad(loss_fn)(net)
        updates, opt_state = optimizer.update(
            grads, opt_state, eqx.filter(net, eqx.is_array)
        )
        net = eqx.apply_updates(net, updates)
        return net, opt_state, loss

    losses = []
    t0 = time.monotonic()
    pbar = tqdm(total=seconds, unit="s", desc="training")
    while time.monotonic() - t0 < seconds:
        key, key_batch = jr.split(key)
        xt, dx, t, y = lisa.get_train_batch(
            key_batch, batch_size=BATCH_SIZE, n_sources=N_SOURCES, t_obs=T_OBS
        )
        x1 = xt + (1.0 - t)[:, None, None] * dx
        net, opt_state, loss = train_step(net, opt_state, y, x1)
        losses.append(loss.item())
        pbar.update(int(time.monotonic() - t0 - pbar.n))
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    pbar.close()

    eqx.tree_serialise_leaves(CHECKPOINT_PATH, net)
    print(f"[checkpoint] saved -> {CHECKPOINT_PATH}")

    plt.figure()
    plt.loglog(losses)
    plt.xlabel("step")
    plt.ylabel("MSE loss")
    plt.grid()
    plt.title("x | y regression loss")
    plt.savefig(LOSS_PLOT_PATH)
    plt.close()

    ########################################
    # evaluation: reconstruct a held-out batch and report MLE accuracy
    key, key_eval = jr.split(key)
    xt, dx, t, y = lisa.get_train_batch(
        key_eval, batch_size=1024, n_sources=N_SOURCES, t_obs=T_OBS
    )
    x1 = xt + (1.0 - t)[:, None, None] * dx
    pred = jax.vmap(lambda yi: net(query, yi, t_fixed))(y)
    pred = jnp.clip(pred, 0.0, 1.0)

    x1_np, pred_np = np.asarray(x1), np.asarray(pred)
    err = np.abs(pred_np - x1_np)  # (B, S, 3) unit-cube abs error

    print("\nUnit-cube MLE reconstruction error (median abs error over sources & batch):")
    for i, name in enumerate(PARAM_NAMES):
        true_i, pred_i = x1_np[..., i], pred_np[..., i]
        r2 = 1 - np.sum((pred_i - true_i) ** 2) / np.sum((true_i - true_i.mean()) ** 2)
        print(f"  {name:6s}  median|err|={np.median(err[..., i]):.4f}   R2={r2:.3f}")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for i, (ax, name) in enumerate(zip(axes, PARAM_NAMES)):
        ax.scatter(x1_np[..., i].ravel(), pred_np[..., i].ravel(), s=3, alpha=0.3)
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel(f"true {name} (unit cube)")
        ax.set_ylabel(f"pred {name}")
        ax.set_title(name)
    fig.suptitle("x | y point-estimate reconstruction (holdout batch)")
    fig.tight_layout()
    fig.savefig(RECON_PLOT_PATH)
    print(f"saved {RECON_PLOT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=1200.0)
    args = parser.parse_args()
    main(args.seconds)
