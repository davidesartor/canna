"""Flow-matching sanity check: 2-source, amplitude-only, low-noise problem.

Extends the single-parameter sanity (regression_sanity_A.py, R2~0.98) to the
generative flow and two sources:

* two galactic binaries at two *distinct fixed frequencies* (so they occupy
  different bins and are identifiable -- same frequency would add coherently
  into one bin and be degenerate),
* each source varies only in amplitude A; f0 (per source), fdot, sky and
  orientation are pinned,
* instrumental noise scaled down by NOISE_SCALE (default 100x quieter),
* the MMDiT flow (same backbone/objective as train.py) learns p(A0, A1 | y).

Trains with conditional flow matching, then samples the posterior for a fixed
injection and corner-plots the two recovered unit-cube amplitudes vs truth.
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
from einops import rearrange
from tqdm import tqdm

jax.config.update("jax_enable_x64", True)

from wdm_transform.transforms import from_freq_to_wdm

from src import lisa, networks, inverse_cdfs

SEED = 0
T_OBS = lisa.MONTH_s
DT = lisa.SAMPLING_STEP_s
NCROP = 32

N_SOURCES = 2
X_DIM = 1  # only amplitude is inferred, per source
A_RANGE = (1e-25, 1.7e-23)
# Two distinct fixed frequencies -> the two sources are separable.
F0_PER_SOURCE = (2.0e-3, 2.5e-3)
FIXED = dict(fdot=1.0e-18, ra=1.0, dec=-0.5,
             psi=float(jnp.pi / 4), iota=1.0, phi0=0.0)
NOISE_SCALE = float(os.environ.get("NOISE_SCALE", "0.01"))

HIDDEN_DIM = 256
NUM_BLOCKS = 4
NUM_HEADS = 8

LEARNING_RATE = 1e-4
BATCH_SIZE = 512
TAG = f"2src_A_noise{NOISE_SCALE:g}"
CHECKPOINT_PATH = f"checkpoint_flow_{TAG}.eqx"
LOSS_PLOT_PATH = f"training_loss_flow_{TAG}.pdf"
CORNER_PATH = f"flow_corner_{TAG}.pdf"


def params_from_uA(u_A):
    """(N_SOURCES,) unit-cube amplitudes -> (N_SOURCES, 8) lisa params."""
    A = inverse_cdfs.log_uniform(u_A, A_RANGE)  # (N_SOURCES,)
    f0 = jnp.array(F0_PER_SOURCE)
    fdot = jnp.full((N_SOURCES,), FIXED["fdot"])
    ra = jnp.full((N_SOURCES,), FIXED["ra"])
    dec = jnp.full((N_SOURCES,), FIXED["dec"])
    psi = jnp.full((N_SOURCES,), FIXED["psi"])
    iota = jnp.full((N_SOURCES,), FIXED["iota"])
    phi0 = jnp.full((N_SOURCES,), FIXED["phi0"])
    return jnp.stack([f0, fdot, A, ra, dec, psi, iota, phi0], axis=-1)


def datastream_to_y(datastream, k_noise):
    noise = lisa.sample_noise(k_noise, t_obs=T_OBS, dt=DT, ncrop=NCROP)
    d = datastream + NOISE_SCALE * noise
    y = from_freq_to_wdm(d.T, nt=32, nf=d.shape[0] // 32,
                         a=1.0 / 3.0, d=1.0, dt=DT, backend="jax")
    y = rearrange(y, "c t f -> t (f c)")
    return jnp.concat([jnp.log(jnp.abs(y)), jnp.sign(y)], axis=-1)


def make_batch(key, batch_size):
    """Conditional flow-matching batch: (xt, dx, t, y)."""
    def one(rng):
        k_x1, k_x0, k_t, k_noise = jr.split(rng, 4)
        x1 = jr.uniform(k_x1, (N_SOURCES, X_DIM))  # unit-cube amplitudes
        x0 = jr.uniform(k_x0, (N_SOURCES, X_DIM))
        t = jr.uniform(k_t)
        signal = lisa.clean_signal(params_from_uA(x1[:, 0]),
                                   t_obs=T_OBS, dt=DT, ncrop=NCROP)
        y = datastream_to_y(signal, k_noise)
        xt = x0 + t * (x1 - x0)
        dx = x1 - x0
        return xt, dx, t, y

    return jax.vmap(one)(jr.split(key, batch_size))


def sample_flow(flow, x0, y, steps=16):
    """RK4 integration t:0->1 (no periodic x%1 wrap); clip to unit cube."""
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
    print(f"NOISE_SCALE={NOISE_SCALE}  f0/source={F0_PER_SOURCE}")
    key = jr.key(SEED)

    key, key_mock = jr.split(key)
    xt, dx, t, y = make_batch(key_mock, 2)
    print(f"xt={xt.shape[1:]}  dx={dx.shape[1:]}  y={y.shape[1:]}")

    key, key_init = jr.split(key)
    flow = networks.MMDiT(
        x_dim=X_DIM, y_dim=y.shape[-1], hidden_dim=HIDDEN_DIM,
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

    # Fixed injection + conditioning for evaluation (built once so the corner
    # plot is comparable across checkpoints).
    u_true = jnp.array([0.7, 0.3])  # unit-cube amplitudes of the two sources
    key, k_noise = jr.split(key)
    eval_signal = lisa.clean_signal(params_from_uA(u_true), t_obs=T_OBS, dt=DT, ncrop=NCROP)
    y_obs = datastream_to_y(eval_signal, k_noise)
    key, k0 = jr.split(key)
    x0_eval = jr.uniform(k0, (3000, N_SOURCES, X_DIM))

    @eqx.filter_jit
    def sample_all(flow, x0, y):
        return jax.vmap(lambda xi: sample_flow(flow, xi, y))(x0)

    def checkpoint_and_eval(flow, losses, elapsed):
        eqx.tree_serialise_leaves(CHECKPOINT_PATH, flow)

        plt.figure()
        plt.loglog(losses)
        plt.xlabel("step"); plt.ylabel("flow-matching MSE"); plt.grid()
        plt.title(f"flow loss (2 src, A only, noise x{NOISE_SCALE:g})")
        plt.savefig(LOSS_PLOT_PATH)
        plt.close()

        samples = np.asarray(sample_all(flow, x0_eval, y_obs))[:, :, 0]  # (N, 2)
        med = np.median(samples, axis=0)
        print(f"[{elapsed/3600:.2f} h] loss={np.mean(losses[-200:]):.5f}  "
              f"injected={np.asarray(u_true)}  post.median={med}", flush=True)

        fig = corner.corner(
            samples, labels=[r"$u_{A_0}$ (f=2.0mHz)", r"$u_{A_1}$ (f=2.5mHz)"],
            truths=np.asarray(u_true), truth_color="black", color="C1",
            range=[(0, 1), (0, 1)], show_titles=True, title_fmt=".3f",
            hist_kwargs={"density": True},
        )
        fig.legend(handles=[
            mlines.Line2D([], [], color="C1", label="flow posterior"),
            mlines.Line2D([], [], color="black", label="injected truth"),
        ], loc="upper right", fontsize=12, frameon=False)
        fig.suptitle(f"2-source amplitude posterior (noise x{NOISE_SCALE:g}, "
                     f"{elapsed/3600:.1f}h)", y=1.02)
        fig.savefig(CORNER_PATH, bbox_inches="tight")
        plt.close(fig)

    # Training loop with periodic checkpoint + eval so a long (e.g. 24h) run is
    # crash-safe and inspectable mid-flight.
    CHECKPOINT_EVERY_s = float(os.environ.get("CHECKPOINT_EVERY_s", "600"))
    losses = []
    t0 = time.monotonic()
    last_save = t0
    pbar = tqdm(total=seconds, unit="s", desc="training")
    while time.monotonic() - t0 < seconds:
        key, key_batch = jr.split(key)
        batch = make_batch(key_batch, BATCH_SIZE)
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
    parser.add_argument("--seconds", type=float, default=1200.0)
    args = parser.parse_args()
    main(args.seconds)
