# plots the distribution of y from lisa.py get_train_batch
import os
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
from einops import rearrange

from src import lisa

SEED = 0
N_SOURCES = int(os.environ.get("N_SOURCES", "1"))
T_OBS = lisa.MONTH_s
N = 100


def robust_hist(data, title, pct=0.0):
    # zoom bins to the bulk of the data (a few extreme outliers otherwise
    # blow up the auto range and squash everything into one bin), and use a
    # log y-axis so the tails/shape are visible instead of just the peak.
    lo, hi = jnp.percentile(data, jnp.array([pct, 100 - pct]))
    plt.hist(data, bins=100, range=(float(lo), float(hi)), density=True)
    plt.yscale("log")
    plt.title(title)


def sym_imshow(data, title):
    vmax = float(jnp.abs(data).max())
    plt.imshow(data, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    plt.title(title)
    plt.colorbar()


if __name__ == "__main__":
    key = jr.key(SEED)

    plt.figure(figsize=(12, 8))

    # plot hist with full noise signal
    _, _, _, y = lisa.get_train_batch(
        key=key,
        batch_size=N,
        n_sources=N_SOURCES,
        t_obs=T_OBS,
        dt=lisa.SAMPLING_STEP_s,
        noise_scale=1.0 / 100,
    )
    y = rearrange(y, "b t (f c) -> (b t f) c", c=3)
    print("y shape", y.shape, "y min", y.min(), "y max", y.max(), y.dtype)

    plt.subplot(2, 3, 1)
    robust_hist(y[:, 0], "A distribution")
    plt.subplot(2, 3, 2)
    robust_hist(y[:, 1], "E distribution")
    plt.subplot(2, 3, 3)
    robust_hist(y[:, 2], "T distribution")

    # plot hist with no noise signal
    _, _, _, y = lisa.get_train_batch(
        key=key,
        batch_size=N,
        n_sources=N_SOURCES,
        t_obs=T_OBS,
        dt=lisa.SAMPLING_STEP_s,
        noise_scale=0.0,
    )
    y = rearrange(y, "b t (f c) -> (b t f) c", c=3)
    print("y shape", y.shape, "y min", y.min(), "y max", y.max(), y.dtype)

    plt.subplot(2, 3, 4)
    robust_hist(y[:, 0], "A distribution (no noise)")
    plt.subplot(2, 3, 5)
    robust_hist(y[:, 1], "E distribution (no noise)")
    plt.subplot(2, 3, 6)
    robust_hist(y[:, 2], "T distribution (no noise)")

    plt.tight_layout()
    plt.savefig("distr_full_noise.png")

    # imshow of one signal (with and without noise)
    # show each of the 3 channels (A/E/T) in a subplot
    plt.figure(figsize=(12, 8))
    _, _, _, y = lisa.get_train_batch(
        key=key,
        batch_size=N,
        n_sources=N_SOURCES,
        t_obs=T_OBS,
        dt=lisa.SAMPLING_STEP_s,
        noise_scale=1.0 / 100,
    )
    y = rearrange(y[0], "t (f c) -> t f c", c=3)
    print("y shape", y.shape, "y min", y.min(), "y max", y.max(), y.dtype)
    for i in range(3):
        plt.subplot(2, 3, i + 1)
        sym_imshow(y[:, :, i], f"Signal with noise (channel {"AET"[i]})")

    _, _, _, y = lisa.get_train_batch(
        key=key,
        batch_size=N,
        n_sources=N_SOURCES,
        t_obs=T_OBS,
        dt=lisa.SAMPLING_STEP_s,
        noise_scale=0.0,
    )
    y = rearrange(y[0], "t (f c) -> t f c", c=3)
    print("y shape", y.shape, "y min", y.min(), "y max", y.max(), y.dtype)
    for i in range(3):
        plt.subplot(2, 3, i + 4)
        sym_imshow(y[:, :, i], f"Signal without noise (channel {"AET"[i]})")

    
    plt.tight_layout()
    plt.savefig("distr_example.png")
