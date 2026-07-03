"""Flow-matching on the amplitude-only, multi-source problem (mirror of
regression_sanity.py).

Same problem as regression_sanity.py -- infer each source's unit-cube
amplitude A from the WDM conditioning y with the 7 other lisa params (f0,
fdot, sky, orientation) varying as random nuisances instead of being pinned --
but trained with the conditional flow-matching objective (like train.py)
instead of a point-estimate regressor.

Eval metric parallels the regression: for each held-out injection, sample the
posterior, take the per-source median as the point estimate, and R2 those
medians against the truths. Also saves a corner plot (A per source) for one
representative injection. Knobs (env): N_SOURCES (default 2), NOISE_SCALE
(default 1.0).
"""
import argparse
import functools
import os
import time

import corner
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import optax
from tqdm import tqdm

jax.config.update("jax_enable_x64", True)

from src import lisa, networks

SEED = 0
N_SOURCES = int(os.environ.get("N_SOURCES", "2"))
T_OBS = lisa.MONTH_s
A_IDX = 2  # index of A in lisa's [f0, fdot, A, ra, dec, psi, iota, phi0]

HIDDEN_DIM = 256
NUM_BLOCKS = 4
NUM_HEADS = 8

LEARNING_RATE = 1e-4
BATCH_SIZE = 512
NOISE_SCALE = lisa.NOISE_SCALE
TAG = f"{N_SOURCES}src_A_noise{NOISE_SCALE:g}"
CHECKPOINT_PATH = f"checkpoint_flow_{TAG}.eqx"
LOSS_PLOT_PATH = f"training_loss_flow_{TAG}.pdf"
CORNER_PATH = f"flow_corner_{TAG}.pdf"
PARAM_NAMES = [f"A_{i}" for i in range(N_SOURCES)]

B_EVAL = 128    # held-out injections for the R2 metric
N_SAMP = 64     # posterior draws per injection


def sample_flow(flow, x0, y, steps=16):
    def body(i, x):
        dt = 1.0 / steps
        t = i * dt
        k1 = flow(x, y, t)
        k2 = flow(x + k1 * dt / 2, y, t + dt / 2)
        k3 = flow(x + k2 * dt / 2, y, t + dt / 2)
        k4 = flow(x + k3 * dt, y, t + dt)
        return x + (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6
    x = jax.lax.fori_loop(0, steps, body, x0)
    return jnp.clip(x, 0.0, 1.0)


def main(seconds: float):
    print(f"JAX backend: {jax.default_backend()}, devices: {jax.local_device_count()}")
    print(f"N_SOURCES={N_SOURCES}  NOISE_SCALE={NOISE_SCALE:g}")
    key = jr.key(SEED)

    key, key_mock = jr.split(key)
    xt, dx, t, y = lisa.get_train_batch(
        key_mock, batch_size=2, n_sources=N_SOURCES, t_obs=T_OBS
    )
    x_dim = 1  # flow-match A only; the other 7 lisa params vary as nuisances
    print(f"x_dim={x_dim}  y_dim={y.shape[-1]}  n_sources={N_SOURCES}  y={y.shape[1:]}")

    key, key_init = jr.split(key)
    flow = networks.MMDiT(
        x_dim=x_dim, y_dim=y.shape[-1], hidden_dim=HIDDEN_DIM,
        num_blocks=NUM_BLOCKS, num_heads=NUM_HEADS, key=key_init,
    )
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(LEARNING_RATE))
    opt_state = optimizer.init(eqx.filter(flow, eqx.is_array))

    @functools.partial(eqx.filter_jit, donate="all")
    def train_step(flow, opt_state, batch):
        xt, dx, t, y = batch

        def loss_fn(flow):
            pred = jax.vmap(flow)(xt, y, t)
            return jnp.mean((pred - dx) ** 2)

        loss, grads = eqx.filter_value_and_grad(loss_fn)(flow)
        updates, opt_state = optimizer.update(
            grads, opt_state, eqx.filter(flow, eqx.is_array)
        )
        flow = eqx.apply_updates(flow, updates)
        return flow, opt_state, loss

    # fixed held-out eval set (truths + conditioning), same each checkpoint
    key, key_eval = jr.split(key)
    xe, dxe, te, ye = lisa.get_train_batch(
        key_eval, batch_size=B_EVAL, n_sources=N_SOURCES, t_obs=T_OBS
    )
    x1_eval = np.asarray(xe + (1.0 - te)[:, None, None] * dxe)[..., A_IDX:A_IDX + 1]  # (B_EVAL, S, 1)
    key, k0 = jr.split(key)
    x0_eval = jr.uniform(k0, (B_EVAL, N_SAMP, N_SOURCES, x_dim))

    @eqx.filter_jit
    def posterior_medians(flow, x0_eval, ye):
        # for each injection: N_SAMP draws -> per-param median
        def one(x0s, y):
            draws = jax.vmap(lambda xi: sample_flow(flow, xi, y))(x0s)  # (N_SAMP,S,D)
            return jnp.median(draws, axis=0)  # (S, D)
        return jax.vmap(one)(x0_eval, ye)  # (B_EVAL, S, D)

    def checkpoint_and_eval(flow, losses, elapsed):
        eqx.tree_serialise_leaves(CHECKPOINT_PATH, flow)

        plt.figure()
        plt.loglog(losses)
        plt.xlabel("step"); plt.ylabel("flow-matching MSE"); plt.grid()
        plt.title(f"flow loss ({TAG})")
        plt.savefig(LOSS_PLOT_PATH); plt.close()

        med = np.asarray(posterior_medians(flow, x0_eval, ye))  # (B_EVAL, S, 1)
        r2s = []
        for i, name in enumerate(PARAM_NAMES):
            tr, pr = x1_eval[:, i, 0], med[:, i, 0]
            r2 = 1 - np.sum((pr - tr) ** 2) / np.sum((tr - tr.mean()) ** 2)
            r2s.append(f"{name} R2={r2:.3f} (|err|={np.median(np.abs(pr-tr)):.3f})")
        print(f"[{elapsed/3600:.2f} h] loss={np.mean(losses[-200:]):.5f}  " + "  ".join(r2s),
              flush=True)

        # corner across sources' amplitudes for one representative injection
        draws0 = np.asarray(
            jax.vmap(lambda xi: sample_flow(flow, xi, ye[0]))(x0_eval[0])
        )[:, :, 0]  # (N_SAMP, S)
        truth0 = x1_eval[0, :, 0]  # (S,)
        fig = corner.corner(
            draws0, labels=[rf"${n}$" for n in PARAM_NAMES], truths=truth0,
            truth_color="black", color="C1", range=[(0, 1)] * N_SOURCES,
            show_titles=True, title_fmt=".3f", hist_kwargs={"density": True},
        )
        fig.legend(handles=[
            mlines.Line2D([], [], color="C1", label="flow posterior"),
            mlines.Line2D([], [], color="black", label="injected truth"),
        ], loc="upper right", fontsize=12, frameon=False)
        fig.suptitle(f"{N_SOURCES}-source amplitude posterior ({TAG}, {elapsed/3600:.1f}h)", y=1.02)
        fig.savefig(CORNER_PATH, bbox_inches="tight"); plt.close(fig)

    CHECKPOINT_EVERY_s = float(os.environ.get("CHECKPOINT_EVERY_s", "300"))
    losses = []
    t0 = time.monotonic()
    last_save = t0
    pbar = tqdm(total=seconds, unit="s", desc="training")
    while time.monotonic() - t0 < seconds:
        key, key_batch = jr.split(key)
        xt, dx, t, y = lisa.get_train_batch(
            key_batch, batch_size=BATCH_SIZE, n_sources=N_SOURCES, t_obs=T_OBS
        )
        batch = xt[..., A_IDX:A_IDX + 1], dx[..., A_IDX:A_IDX + 1], t, y
        flow, opt_state, loss = train_step(flow, opt_state, batch)
        losses.append(loss.item())
        pbar.update(int(time.monotonic() - t0 - pbar.n))
        pbar.set_postfix(loss=f"{loss.item():.5f}")
        if time.monotonic() - last_save >= CHECKPOINT_EVERY_s:
            checkpoint_and_eval(flow, losses, time.monotonic() - t0)
            last_save = time.monotonic()
    pbar.close()

    checkpoint_and_eval(flow, losses, time.monotonic() - t0)
    print(f"[checkpoint] final saved -> {CHECKPOINT_PATH}")
    print(f"saved {CORNER_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=1800.0)
    args = parser.parse_args()
    main(args.seconds)
