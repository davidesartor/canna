"""Figure: closest f0 separation per training window, whole band and its five subbands, against the two-source pairs of Delmond et al."""

import os
import re

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
BAND_NAMES = ("LowTail", "LowFreq", "Core", "HighFreq", "HighTail")
BAND_LABELS = ("low tail", "low freq", "core", "high freq", "high tail")


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


# each subband of numbers/narrow-band.tex at its own R over its own f0 range, inside the full run's problem
full_problem = config.load("full-B").problem
band_macros = dict(
    re.findall(r"\\newcommand\{\\NarrowBand(\w+)\}\{([^}]*)\}", (style.NUMBERS / "narrow-band.tex").read_text())
)
problems = {"whole band": LisaGB(**full_problem)} | {
    label: LisaGB(
        **full_problem
        | dict(
            response_points=int(band_macros[f"{name}R"]),
            f0_range=(float(band_macros[f"{name}LowMhz"]) * 1e-3, float(band_macros[f"{name}HighMhz"]) * 1e-3),
        )
    )
    for name, label in zip(BAND_NAMES, BAND_LABELS)
}
closest = {band: closest_pair_bins(p, jr.key(0)) for band, p in problems.items()}

style.setup()
fig, ax = plt.subplots(figsize=(style.COLUMN, 2.2), constrained_layout=True)
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
        zorder=4,
        bbox=dict(facecolor="white", edgecolor="none", pad=0.6),
    )
# the whole band in ink, its subbands from light (low f0) to dark (high f0)
colours = (style.INK, *style.BLUE_RAMP[1:4], *style.BLUE_RAMP[5:7])
for (band, gaps), colour in zip(closest.items(), colours):
    gaps = np.sort(gaps)
    ax.step(
        gaps,
        np.arange(1, gaps.size + 1) / gaps.size,
        where="post",
        color=colour,
        lw=1.4 if band == "whole band" else 0.9,
        zorder=3 if band == "whole band" else 2,
        label=band,
    )
ax.legend(loc="lower right", fontsize=5.8, frameon=False, handlelength=1.4, labelspacing=0.25)
ax.set_xscale("log")
ax.set_xlim(0.1, 4e3)
ax.set_ylim(0, 1)
ax.set_xlabel(r"closest $f_0$ separation in a window [bins of $1/T_{\rm obs}$]")
ax.set_ylabel("fraction of windows")
style.save(fig, "crowding")
