"""figures/priors.pdf: the prior over a window, from the closed forms and the drawn sources, and where training centres its windows."""

import json
import math

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

import style

from canna.lisa import config
from canna.lisa.physics import (
    SOURCE_COVARIANCE,
    SOURCE_MEAN,
    galactic_to_equatorial,
    sky_log_density,
)
from canna.lisa.problem import LisaGB

BANDS = {"narrow": style.ORANGE, "full": style.BLUE}
SKY_LEVELS = (0.99, 0.9, 0.5)
VISIBLE_SNR = 7.0
SCATTER_POINTS = 6000
F0_MEAN, F0_STD = SOURCE_MEAN[0], math.sqrt(SOURCE_COVARIANCE[0, 0])


style.setup()
problems = {band: LisaGB(**config.load(f"{band}-B").problem) for band in BANDS}
data = style.REPO / "reports/prior/f0-fdot-amplitude/data"
report = json.loads((data / "f0-fdot-amplitude.json").read_text())
prior, curves = report["prior"], report["bands"]["full"]["curves"]
drawn = {band: np.load(data / f"{band}.npz") for band in BANDS}
cut = {band: np.load(data / f"{band}-cut.npz") for band in BANDS}
full, t_obs = drawn["full"], problems["full"].t_obs
visible = full["snr"].ravel() >= VISIBLE_SNR
rng = np.random.default_rng(0)
shown = rng.choice(visible.size, SCATTER_POINTS, replace=False)
narrow_top = math.log10(problems["narrow"].f0_range[1])
log_f0_curve = np.asarray(curves["log_f0"])
fig, axes = plt.subplots(2, 3, figsize=(style.FULL, 3.6), layout="constrained")


def sources(ax, y: np.ndarray) -> None:
    """The full band's drawn sources at y against log10 f0, a sample of them muted and every visible one dark."""
    log_f0 = full["log_f0"].ravel()
    ax.scatter(
        log_f0[shown], y[shown], color=style.BLUE_RAMP[1], s=1.2, lw=0, rasterized=True
    )
    ax.scatter(
        log_f0[visible],
        y[visible],
        color=style.BLUE_RAMP[5],
        s=2.0,
        lw=0,
        rasterized=True,
        label=rf"SNR $\geq$ {VISIBLE_SNR:.0f}",
    )
    ax.axvline(narrow_top, **style.GUIDE)


def mhz_axis(ax) -> None:
    ticks = [0.1, 0.3, 1, 3, 10]
    ax.set_xticks(np.log10(np.asarray(ticks) * 1e-3), [f"{t:g}" for t in ticks])
    ax.set(
        xlabel=r"$f_0$ [mHz]",
        xlim=(-4.05, math.log10(problems["full"].f0_range[1]) + 0.05),
    )


# (a) f0: each band's drawn sources against the catalogue marginal truncated to the band, and where the visible ones lie
ax = axes[0, 0]
for band, color in reversed(BANDS.items()):
    lo, hi = (math.log10(f) for f in problems[band].f0_range)
    edges = np.linspace(lo, hi, 41 if band == "full" else 21)
    ax.stairs(
        np.histogram(drawn[band]["log_f0"].ravel(), edges, density=True)[0],
        edges,
        color=color,
        fill=True,
        alpha=0.3,
        lw=0,
    )
    x = np.linspace(lo, hi, 400)
    density = norm.pdf(x, F0_MEAN, F0_STD) / (
        norm.cdf(hi, F0_MEAN, F0_STD) - norm.cdf(lo, F0_MEAN, F0_STD)
    )
    ax.plot(x, density, color=color, lw=1.0, label=band)
edges = np.linspace(-4, math.log10(problems["full"].f0_range[1]), 41)
ax.stairs(
    np.histogram(full["log_f0"].ravel()[visible], edges, density=True)[0],
    edges,
    color=style.INK,
    lw=0.8,
    ls=(0, (2, 1.2)),
    label=rf"full, SNR $\geq$ {VISIBLE_SNR:.0f}",
)
mhz_axis(ax)
ax.set(ylabel=r"$p(\log_{10} f_0)$")
ax.set_ylim(bottom=0)
ax.legend(loc="upper left")
style.panel_label(ax, "a")

# (b) sky: highest-density regions per steradian, in the equal-area (l, sin b)
ax = axes[0, 1]
l = np.linspace(-math.pi, math.pi, 721)
sin_b = np.linspace(-1, 1, 1601)[1:-1]
L, SB = np.meshgrid(l, sin_b)
ra, dec = galactic_to_equatorial(jnp.asarray(L), jnp.arcsin(jnp.asarray(SB)))
density = np.exp(np.asarray(sky_log_density(ra, dec))) / np.cos(np.asarray(dec))
order = np.sort(density.ravel())[::-1]
mass = np.cumsum(order) / order.sum()
levels = [order[np.searchsorted(mass, q)] for q in SKY_LEVELS]
ax.pcolormesh(
    np.degrees(L), SB, np.log10(density), cmap=style.SEQUENTIAL_CMAP, rasterized=True
)
contours = ax.contour(
    np.degrees(L), SB, density, levels=levels, colors=style.INK, linewidths=0.5
)
ax.clabel(
    contours,
    fmt={level: f"{q:.0%}" for level, q in zip(levels, SKY_LEVELS)},
    fontsize=5.5,
    inline_spacing=2,
)
ax.grid(False)
ax.set(
    xlim=(180, -180),
    ylim=(-1, 1),
    xticks=[180, 90, 0, -90, -180],
    xlabel=r"galactic longitude $l$ [deg]",
    ylabel=r"$\sin b$",
)
style.panel_label(ax, "b")

# (c) chirp drift over the observation in bins: the conditional mean and its residual band, one bin dashed
ax = axes[0, 2]
sources(ax, np.log10(full["drift"].ravel()))
drift_mean = np.asarray(curves["fdot_mean"]) + math.log10(t_obs**2)
residual = prior["residual_std"][0]
ax.fill_between(
    log_f0_curve,
    drift_mean - residual,
    drift_mean + residual,
    color=style.FILL,
    zorder=1.5,
)
ax.plot(log_f0_curve, drift_mean, color=style.INK, lw=1.0)
ax.axhline(0, **style.GUIDE)
mhz_axis(ax)
ax.set(ylabel=r"$\log_{10}\dot f\,T_{\rm obs}^2$ [bins]")
ax.legend(loc="upper left", markerscale=3)
style.panel_label(ax, "c")

# (d) where the window centres fall: the catalogue's f0 marginal against training's visible prior, per band
ax = axes[1, 0]
for band, color in reversed(BANDS.items()):
    lo, hi = (math.log10(f) for f in problems[band].f0_range)
    edges = np.linspace(lo, hi, 41 if band == "full" else 21)
    (k,) = np.flatnonzero(cut[band]["visible_snr"] == VISIBLE_SNR)
    ax.stairs(
        np.histogram(np.log10(drawn[band]["f"]), edges, density=True)[0],
        edges,
        color=color,
        lw=0.8,
        ls=(0, (2, 1.2)),
        label=f"{band}, catalogue",
    )
    ax.stairs(
        np.histogram(np.log10(cut[band]["f"][k]), edges, density=True)[0],
        edges,
        color=color,
        lw=1.0,
        label=f"{band}, training",
    )
mhz_axis(ax)
ax.set(ylabel=r"$p(\log_{10} f_{\rm w})$")
ax.set_ylim(bottom=0)
ax.legend(loc="upper left", handlelength=1.8)
style.panel_label(ax, "d")

# (e) amplitude: the conditional mean, and the amplitude at which an average source reaches SNR 7
ax = axes[1, 1]
sources(ax, full["log_a"].ravel())
amp_mean = np.asarray(curves["amp_mean"])
residual = prior["residual_std"][1]
ax.fill_between(
    log_f0_curve, amp_mean - residual, amp_mean + residual, color=style.FILL, zorder=1.5
)
ax.plot(log_f0_curve, amp_mean, color=style.INK, lw=1.0, label="conditional mean")
ax.plot(
    log_f0_curve,
    curves["amp_snr7"],
    color=style.INK,
    lw=0.8,
    ls=(0, (2, 1.2)),
    label=rf"SNR {VISIBLE_SNR:.0f}, sky average",
)
mhz_axis(ax)
ax.set(ylabel=r"$\log_{10}\mathcal{A}$")
ax.legend(loc="upper right", markerscale=3)
style.panel_label(ax, "e")

# (f) SNR of a source alone in its window, per band
ax = axes[1, 2]
edges = np.linspace(-4, 2.5, 66)
for band, color in reversed(BANDS.items()):
    ax.stairs(
        np.histogram(np.log10(drawn[band]["snr"].ravel()), edges, density=True)[0],
        edges,
        color=color,
        lw=1.0,
        label=band,
    )
ax.axvline(math.log10(VISIBLE_SNR), **style.GUIDE)
ax.set_xticks(
    [-3, -2, -1, 0, 1, 2, 3], ["0.001", "0.01", "0.1", "1", "10", "100", "1000"]
)
ax.set(xlabel=r"SNR $\rho$", ylabel=r"$p(\log_{10}\rho)$")
ax.set_ylim(bottom=0)
ax.legend(loc="upper right")
style.panel_label(ax, "f")

style.save(fig, "priors")
