"""figures/priors.pdf: the prior over a window, from the closed forms and from the drawn prior sources."""

import json
import math

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta

import style

from canna.lisa import config
from canna.lisa.physics import (
    chirp_mass_log_density,
    fdot_from_chirp_mass,
    galactic_to_equatorial,
    sky_log_density,
)
from canna.lisa.problem import WINDOW_STRIDE, LisaGB

BANDS = {"narrow": style.ORANGE, "full": style.BLUE}
SKY_LEVELS = (0.99, 0.9, 0.5)
SLOT_SHADES = (6, 5, 3, 2)


def f0_marginal_ratio(problem: LisaGB, grid: np.ndarray) -> np.ndarray:
    """Exact p(ln f0) of sample_f + f0_window over the log-uniform density, on a grid of ln(f0 T)."""
    low, high = (f * problem.t_obs for f in problem.f0_range)
    span = math.log(high / low)
    bins, guard = problem.window_bins, problem.response_points // 2
    lo, hi = problem.window_index_range
    index = np.arange(lo, hi + 1)
    centre = index * WINDOW_STRIDE + bins / 2
    u_of = lambda c: np.clip(np.log(np.maximum(c, 1e-300) / low) / span, 0.0, 1.0)
    upper = np.where(index == hi, np.inf, centre + WINDOW_STRIDE / 2)
    lower = np.where(index == lo, 0.0, centre - WINDOW_STRIDE / 2)
    prob = u_of(upper) - u_of(lower)
    a = np.log(np.maximum(index * WINDOW_STRIDE + guard, low))
    b = np.log(np.minimum(index * WINDOW_STRIDE + bins - guard, high))
    height = prob / (b - a)
    density = np.zeros(grid.size + 1)
    np.add.at(density, np.searchsorted(grid, a), height)
    np.add.at(density, np.searchsorted(grid, b, side="right"), -height)
    return np.cumsum(density)[:-1] * span


def per_unit(hist: dict, scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """A recorded histogram as a density per unit of its axis times scale (edges, density)."""
    edges, counts = np.asarray(hist["edges"]), np.asarray(hist["counts"])
    total = counts.sum() + hist["outside"]
    return edges * scale, counts / total / ((edges[1] - edges[0]) * scale)


style.setup()
problems = {band: LisaGB(**config.load(f"{band}-B").problem) for band in BANDS}
problem = problems["full"]
report = json.loads((style.REPO / "reports/prior/data/prior.json").read_text())["bands"]
fig, axes = plt.subplots(2, 3, figsize=(style.FULL, 3.6), layout="constrained")

# (a) chirp mass: p(ln Mc) by quadrature against the drawn histogram, median marked
ax = axes[0, 0]
log_mc, log_p = chirp_mass_log_density(problem.mass_range)
p = np.exp(log_p)
cdf = np.cumsum(p) / p.sum()
edges, density = per_unit(report["full"]["chirp_mass_check"])
ax.stairs(
    density, np.exp(edges), color=style.BLUE_RAMP[2], fill=True, lw=0, label="drawn"
)
ax.plot(np.exp(log_mc), p, color=style.INK, lw=1.0, label="quadrature")
ax.axvline(math.exp(np.interp(0.5, cdf, log_mc)), **style.GUIDE)
ax.set(
    xscale="log", xlabel=r"$\mathcal{M}_c$ [$M_\odot$]", ylabel=r"$p(\ln\mathcal{M}_c)$"
)
ax.set_xticks([0.2, 0.5, 1.0], ["0.2", "0.5", "1"])
ax.xaxis.set_minor_formatter(plt.NullFormatter())
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

# (c) per-slot SNR: the k-th loudest of S log-uniform draws, closed form over the drawn histograms
ax = axes[0, 2]
S, (snr_lo, snr_hi) = problem.n_sources, problem.snr_range
rho = np.geomspace(snr_lo, snr_hi, 400)
u = np.log(rho / snr_lo) / math.log(snr_hi / snr_lo)
for k, shade in zip(range(1, S + 1), SLOT_SHADES):
    color = style.BLUE_RAMP[shade]
    edges, drawn = per_unit(report["full"]["snr_rank"][k - 1], math.log(10))
    ax.stairs(drawn, np.exp(edges), color=color, fill=True, alpha=0.35, lw=0)
    slot_density = beta.pdf(u, S - k + 1, k) / math.log(snr_hi / snr_lo)
    ax.plot(rho, slot_density, color=color, lw=1.0)
    peak = np.clip(np.argmax(slot_density), 40, rho.size - 40)
    ax.annotate(
        f"{k}",
        (rho[peak], slot_density[peak]),
        xytext=(0, 2),
        textcoords="offset points",
        ha="center",
        va="bottom",
        color=color,
        fontsize=6.5,
    )
edges, window = per_unit(report["full"]["window_snr_hist"], math.log(10))
ax.stairs(
    window,
    np.exp(edges),
    color=style.INK,
    lw=0.8,
    ls=(0, (2, 1.2)),
    label="whole window",
)
ax.axhline(1 / math.log(snr_hi / snr_lo), **style.GUIDE)
ax.set(
    xscale="log",
    xlabel=r"SNR $\rho$",
    ylabel=r"$p(\ln\rho)$",
    xlim=(snr_lo, 2 * snr_hi),
)
ax.set_ylim(0, 0.95)
ax.legend(loc="upper left")
style.panel_label(ax, "c")

# (d) the f0 marginal over the log-uniform density near the band edges, per band
ax = axes[1, 0]
distance = np.linspace(0, 6, 1201)
for band, color in BANDS.items():
    problem = problems[band]
    low, high = (f * problem.t_obs for f in problem.f0_range)
    R = problem.response_points
    bottom = f0_marginal_ratio(problem, np.log(low + R * distance))
    top = f0_marginal_ratio(problem, np.log(high - R * distance)[::-1])[::-1]
    ax.plot(distance, bottom, color=color, label=f"{band}, bottom")
    ax.plot(distance, top, color=color, ls=(0, (3, 1.5)), lw=1.0, label=f"{band}, top")
ax.axhline(1, **style.GUIDE)
ax.set(
    xlabel=r"distance from the band edge [$R$ bins]",
    ylabel=r"$p(\ln f_0)$ / log-uniform",
    xlim=(0, 6),
    ylim=(0.3, 1.6),
)
ax.legend(loc="lower right", handlelength=1.8, ncols=2, columnspacing=0.8)
style.panel_label(ax, "d")

# (e) amplitude against f0: the drawn sources inside the sky-averaged band of the SNR range
ax = axes[1, 1]
for band, color in reversed(BANDS.items()):
    scatter, band_edges = report[band]["scatter"], report[band]["amplitude_band"]
    ax.scatter(
        1e3 * np.asarray(scatter["f0"]),
        scatter["A"],
        color=color,
        s=1.5,
        lw=0,
        alpha=0.6,
        rasterized=True,
    )
    for side in ("lo", "hi"):
        ax.plot(
            1e3 * np.asarray(band_edges["f0"]),
            band_edges[side],
            color=color,
            lw=0.8,
            ls="--",
        )
ax.set(
    xscale="log", yscale="log", xlabel=r"$f_0$ [mHz]", ylabel=r"amplitude $\mathcal{A}$"
)
style.panel_label(ax, "e")

# (f) chirp drift over the observation in frequency bins against f0
ax = axes[1, 2]
for band, color in reversed(BANDS.items()):
    scatter = report[band]["scatter"]
    f0, mc = np.asarray(scatter["f0"]), np.asarray(scatter["mc"])
    drift_bins = np.asarray(fdot_from_chirp_mass(mc, f0)) * problems[band].t_obs ** 2
    ax.scatter(
        1e3 * f0,
        drift_bins,
        color=color,
        s=1.5,
        lw=0,
        alpha=0.6,
        rasterized=True,
        label=band,
    )
ax.axhline(1, **style.GUIDE)
ax.set(
    xscale="log",
    yscale="log",
    xlabel=r"$f_0$ [mHz]",
    ylabel=r"chirp drift $\dot f\,T_{\rm obs}^2$ [bins]",
)
ax.legend(loc="upper left", markerscale=3)
style.panel_label(ax, "f")

style.save(fig, "priors")
