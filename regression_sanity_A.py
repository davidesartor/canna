"""Simplified sanity check: can the network read off a *single* parameter (A)?

This is the easiest possible version of "is the WDM conditioning y digestible":

* one source,
* only the amplitude A varies; every other parameter (f0, fdot, sky,
  orientation) is held fixed, so the datastream depends on A alone (plus noise),
* the instrumental noise is scaled down by NOISE_SCALE (default 100x quieter),
* a point-estimate regressor (same MMDiT backbone as train.py) must recover the
  unit-cube amplitude u_A in [0, 1] from y.

If R2 here is still ~0 the conditioning encoding itself is broken; if it's high
the earlier failure was the multi-parameter / label-switching difficulty, not y.
The WDM encoding replicates lisa.get_train_batch exactly (concat[log|y|, sign]).
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
from einops import rearrange
from tqdm import tqdm

jax.config.update("jax_enable_x64", True)

from wdm_transform.transforms import from_freq_to_wdm

from src import lisa, networks, inverse_cdfs

SEED = 0
T_OBS = lisa.MONTH_s
DT = lisa.SAMPLING_STEP_s
NCROP = 32

# Only A is inferred; everything else is pinned.
A_RANGE = (1e-25, 1.7e-23)  # == lisa.prior_inverse_cdf amplitude prior
FIXED = dict(f0=2.0e-3, fdot=1.0e-18, ra=1.0, dec=-0.5,
             psi=float(jnp.pi / 4), iota=1.0, phi0=0.0)
NOISE_SCALE = float(os.environ.get("NOISE_SCALE", "0.01"))  # 100x quieter default

# model (same size as train.py's MMDiT)
HIDDEN_DIM = 256
NUM_BLOCKS = 4
NUM_HEADS = 8

LEARNING_RATE = 1e-4
BATCH_SIZE = 512
TAG = f"A_noise{NOISE_SCALE:g}"
CHECKPOINT_PATH = f"checkpoint_regression_{TAG}.eqx"
LOSS_PLOT_PATH = f"training_loss_regression_{TAG}.pdf"
RECON_PLOT_PATH = f"regression_reconstruction_{TAG}.pdf"


def make_batch(key, batch_size):
    """(x1, y): x1 = unit-cube amplitude (B,1,1); y = WDM image (B,T,C)."""
    def one(rng):
        k_A, k_noise = jr.split(rng)
        u_A = jr.uniform(k_A)
        A = inverse_cdfs.log_uniform(u_A, A_RANGE)
        params = jnp.array([[FIXED["f0"], FIXED["fdot"], A, FIXED["ra"],
                             FIXED["dec"], FIXED["psi"], FIXED["iota"],
                             FIXED["phi0"]]])
        signal = lisa.clean_signal(params, t_obs=T_OBS, dt=DT, ncrop=NCROP)
        noise = lisa.sample_noise(k_noise, t_obs=T_OBS, dt=DT, ncrop=NCROP)
        datastream = signal + NOISE_SCALE * noise

        # WDM conditioning — identical recipe to lisa.get_train_batch.
        y = from_freq_to_wdm(
            datastream.T, nt=32, nf=datastream.shape[0] // 32,
            a=1.0 / 3.0, d=1.0, dt=DT, backend="jax",
        )
        y = rearrange(y, "c t f -> t (f c)")
        y = jnp.concat([jnp.log(jnp.abs(y)), jnp.sign(y)], axis=-1)

        x1 = u_A[None, None]  # (1 source, 1 param)
        return x1, y

    return jax.vmap(one)(jr.split(key, batch_size))


def main(seconds: float):
    print(f"JAX backend: {jax.default_backend()}, devices: {jax.local_device_count()}")
    print(f"NOISE_SCALE={NOISE_SCALE}  (noise amplitude x{NOISE_SCALE:g})")
    key = jr.key(SEED)

    key, key_mock = jr.split(key)
    x1, y = make_batch(key_mock, 2)
    x_dim = x1.shape[-1]
    print(f"x_dim={x_dim}  y_dim={y.shape[-1]}  n_sources={x1.shape[1]}  y={y.shape[1:]}")

    key, key_init = jr.split(key)
    net = networks.MMDiT(
        x_dim=x_dim, y_dim=y.shape[-1], hidden_dim=HIDDEN_DIM,
        num_blocks=NUM_BLOCKS, num_heads=NUM_HEADS, key=key_init,
    )
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(LEARNING_RATE))
    opt_state = optimizer.init(eqx.filter(net, eqx.is_array))

    query = jnp.zeros((1, x_dim))
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
        x1, y = make_batch(key_batch, BATCH_SIZE)
        net, opt_state, loss = train_step(net, opt_state, y, x1)
        losses.append(loss.item())
        pbar.update(int(time.monotonic() - t0 - pbar.n))
        pbar.set_postfix(loss=f"{loss.item():.5f}")
    pbar.close()

    eqx.tree_serialise_leaves(CHECKPOINT_PATH, net)
    print(f"[checkpoint] saved -> {CHECKPOINT_PATH}")

    plt.figure()
    plt.loglog(losses)
    plt.xlabel("step"); plt.ylabel("MSE loss"); plt.grid()
    plt.title(f"A | y regression loss (noise x{NOISE_SCALE:g})")
    plt.savefig(LOSS_PLOT_PATH)
    plt.close()

    # evaluation on a held-out batch
    key, key_eval = jr.split(key)
    x1, y = make_batch(key_eval, 2048)
    pred = jax.vmap(lambda yi: net(query, yi, t_fixed))(y)
    pred = jnp.clip(pred, 0.0, 1.0)
    true = np.asarray(x1[:, 0, 0])
    pred = np.asarray(pred[:, 0, 0])
    r2 = 1 - np.sum((pred - true) ** 2) / np.sum((true - true.mean()) ** 2)
    print(f"\nA reconstruction (unit cube):  median|err|={np.median(np.abs(pred-true)):.4f}"
          f"   R2={r2:.3f}")

    plt.figure(figsize=(5, 5))
    plt.scatter(true, pred, s=4, alpha=0.3)
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("true u_A (unit cube)"); plt.ylabel("pred u_A")
    plt.title(f"A | y reconstruction (noise x{NOISE_SCALE:g}, R2={r2:.3f})")
    plt.tight_layout()
    plt.savefig(RECON_PLOT_PATH)
    print(f"saved {RECON_PLOT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=1200.0)
    args = parser.parse_args()
    main(args.seconds)
