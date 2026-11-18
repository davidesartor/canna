"""Figures: one corner per band and source count (corner-<band>-<n>.pdf), flow, joint MCMC and Fisher on the closest loud
pair of one eval window (axes: 2-98% of flow and MCMC), from data/corners/<band>-<n>.npz (import_corners.py). Offsets from the truth, sources relabelled by
f0 order for display; f0, ln A and right ascension of both sources. MCMC filled, flow solid, Fisher dashed; 68% and 95%."""

import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter

import style
from style import plt

DATA = style.PAPER / "data/corners"
CONFIGS = [("narrow", 2), ("narrow", 4), ("narrow", 8), ("full", 4), ("full", 8), ("full", 16)]
COLUMNS = [0, 2, 3]  # of mcmc.COLUMNS: f0 [bins], log A, lon (equatorial)
NAMES = [r"\Delta f_0", r"\Delta\ln\mathcal{A}", r"\Delta\alpha"]
UNITS = [" [bins]", "", " [rad]"]
MASSES = (0.95, 0.68)
BINS = 36
SETS = [("mcmc", style.BLUE, True, "-"), ("flow", style.ORANGE, False, "-"), ("fisher", style.MUTED, False, "--")]


def credible_levels(density, masses):
    ordered = np.sort(density.ravel())[::-1]
    cumulative = np.cumsum(ordered) / ordered.sum()
    return sorted({ordered[min(np.searchsorted(cumulative, m), len(ordered) - 1)] for m in masses})


def contours(ax, x, y, extent, filled, color, ls):
    hist, xe, ye = np.histogram2d(x, y, bins=BINS, range=extent)
    density = gaussian_filter(hist.T, 1.2)
    if density.max() <= 0:
        return
    xc, yc = (xe[1:] + xe[:-1]) / 2, (ye[1:] + ye[:-1]) / 2
    levels = credible_levels(density, MASSES)
    if len(levels) < 2:
        return
    if filled:
        ax.contourf(xc, yc, density, levels=[*levels, density.max() * 1.01],
                    colors=[to_rgba(color, 0.3), to_rgba(color, 0.75)])
    else:
        ax.contour(xc, yc, density, levels=levels, colors=color, linewidths=[0.55, 0.95], linestyles=ls)


def extent_of(sets, c):
    """The 2-98% range of the flow and the MCMC together: a few far-tail draws (a neighbour's mode) fall outside."""
    lo = min(np.quantile(s[:, c], 0.02) for s in sets)
    hi = max(np.quantile(s[:, c], 0.98) for s in sets)
    pad = 0.05 * (hi - lo) or 1e-6
    return lo - pad, hi + pad


style.setup()
for band, n in CONFIGS:
    if not (DATA / f"{band}-{n}.npz").is_file():
        # the e64 model has not landed (full S=4): a placeholder, so no older model's corner stays in the paper
        fig = plt.figure(figsize=(style.FULL, 1.2))
        fig.text(0.5, 0.5, f"{band} band, S = {n}: 64-epoch model pending", ha="center", va="center",
                 color=style.MUTED)
        style.save(fig, f"corner-{band}-{n}")
        continue
    d = np.load(DATA / f"{band}-{n}.npz")
    dims = [s * 8 + c for s in (0, 1) for c in COLUMNS]
    labels = [rf"${NAMES[i]}^{{({s + 1})}}${UNITS[i]}" for s in (0, 1) for i in range(len(COLUMNS))]
    sets = {name: d[name][:, dims] for name, *_ in SETS}
    k = len(dims)
    fig, axes = plt.subplots(k, k, figsize=(style.FULL, style.FULL * 0.86), layout="constrained")
    ext = [extent_of([sets["flow"], sets["mcmc"]], c) for c in range(k)]
    for r in range(k):
        for c in range(k):
            ax = axes[r, c]
            if c > r:
                ax.axis("off")
                continue
            if r == c:
                for name, color, filled, ls in SETS:
                    ax.hist(sets[name][:, c], bins=BINS, range=ext[c], density=True, histtype="step",
                            color=color, lw=0.9, ls=ls)
                    if filled:
                        ax.hist(sets[name][:, c], bins=BINS, range=ext[c], density=True, color=to_rgba(color, 0.2))
                ax.set_yticks([])
                ax.axvline(0, **style.GUIDE)
            else:
                for name, color, filled, ls in SETS:
                    contours(ax, sets[name][:, c], sets[name][:, r], [ext[c], ext[r]], filled, color, ls)
                ax.axvline(0, **style.GUIDE)
                ax.axhline(0, **style.GUIDE)
                ax.set_ylim(ext[r])
            ax.set_xlim(ext[c])
            ax.tick_params(labelsize=5, length=2, pad=1)
            if r < k - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel(labels[c], fontsize=7)
            if c > 0 or r == 0:
                ax.set_yticklabels([]) if r != c else None
            else:
                ax.set_ylabel(labels[r], fontsize=7)
    snr = d["snr"]
    fig.legend(
        handles=[Line2D([], [], color=color, lw=1.0, ls=ls) for _, color, _, ls in SETS],
        labels=["joint MCMC", "flow", "Fisher"], loc="upper right", frameon=False, fontsize=7,
        title=(f"{band} band, S = {n}: injection {int(d['injection'])}\n"
               f"sources at SNR {snr[0]:.0f} and {snr[1]:.0f}, {float(d['sep_bins']):.2f} bins apart"),
        title_fontsize=7, bbox_to_anchor=(0.98, 0.98),
    )
    style.save(fig, f"corner-{band}-{n}")
    plt.close(fig)
    print(f"corner-{band}-{n}.pdf")
