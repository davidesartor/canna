"""Figure: closest f0 separation per training window, narrow and full band, against the two-source pairs of Delmond et al."""

import os

import numpy as np

import style
from style import plt

import jax
import jax.numpy as jnp
import jax.random as jr
from canna.lisa import config
from canna.lisa.physics import YEAR
from canna.lisa.problem import LisaGB
from canna.lisa.simulator import sample_physical, sample_window

assert not [k for k in os.environ if k.startswith("LISA_")], "stray LISA_* env vars"

# Delmond et al. 2026: two sources at 1 yr, close to MCMC at 0.5 uHz, broad and biased at 0.2 uHz
DELMOND_TOBS, DELMOND_PAIRS_DF0 = YEAR, {
    "broad, biased": 0.2e-6,
    "close to MCMC": 0.5e-6,
}
CROWDING_WINDOWS, CHUNK = 4096, 256


def closest_pair_bins(problem: LisaGB, key) -> np.ndarray:
    """Closest f0 separation per training window, in the analysis's own 1/t_obs bins."""

    def one(key):
        key_f, key_p = jr.split(key)
        f = sample_window(problem, key_f)
        bins = sample_physical(problem, key_p, f)[:, 0] * problem.t_obs
        gaps = jnp.abs(bins[:, None] - bins[None, :])
        return jnp.min(jnp.where(jnp.eye(problem.n_sources, dtype=bool), jnp.inf, gaps))

    keys = jr.split(key, CROWDING_WINDOWS).reshape(-1, CHUNK)
    batched = jax.jit(jax.vmap(one))
    return np.concatenate([np.asarray(batched(chunk)) for chunk in keys])


closest = {
    band: closest_pair_bins(LisaGB(**config.load(f"{band}-B").problem), jr.key(0))
    for band in ("narrow", "full")
}

style.setup()
fig, ax = plt.subplots(figsize=(style.COLUMN, 1.9), constrained_layout=True)
ax.axvspan(1e-3, 1.0, color=style.GRID, lw=0, zorder=0)
ax.text(
    0.32,
    0.97,
    "within\none bin",
    ha="center",
    va="top",
    color=style.MUTED,
    fontsize=6,
    transform=ax.get_xaxis_transform(),
)
for (label, df0), ls in zip(DELMOND_PAIRS_DF0.items(), ("-", "--")):
    bins = df0 * DELMOND_TOBS
    ax.axvline(bins, color=style.INK2, lw=0.7, ls=ls, zorder=1)
    ax.text(
        bins * 1.08,
        0.14,
        label,
        rotation=90,
        ha="left",
        va="bottom",
        color=style.INK2,
        fontsize=6,
        transform=ax.get_xaxis_transform(),
    )
# label each curve in the empty region above it
for (band, gaps), colour, label_x in zip(
    closest.items(), (style.BLUE, style.ORANGE), (1.3, 70.0)
):
    gaps = np.sort(gaps)
    ax.step(
        gaps,
        np.arange(1, gaps.size + 1) / gaps.size,
        where="post",
        color=colour,
        zorder=2,
    )
    ax.text(label_x, 0.62, f"{band} band", ha="center", color=colour)
ax.set_xscale("log")
ax.set_xlim(0.1, 4e3)
ax.set_ylim(0, 1)
ax.set_xlabel(r"closest $f_0$ separation in a window [bins of $1/T_{\rm obs}$]")
ax.set_ylabel("fraction of windows")
style.save(fig, "crowding")
