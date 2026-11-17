"""Figure: Fisher widths against the joint reference MCMC, binned in source SNR, per band and source count.

updated 2026-11-18: from the six packed references in use (mcmc_reference.py) instead of the old eight joint chains.
Per SNR bin, the median over sources of Fisher / MCMC width: f0 (solid) and the median over the six parameters other
than f0 and fdot (dashed); bins with fewer than MIN_SOURCES sources are left out."""

import numpy as np

import mcmc_reference
import style
from style import plt

FISHER_BAND = (0.8, 1.25)
SNR_EDGES = np.array([1, 2, 3, 5, 7, 10, 15, 25, 40, 70, 150, 400])
MIN_SOURCES = 5
COLORS = {2: style.BLUE, 4: style.BLUE, 8: style.ORANGE, 16: style.INK2}  # S = 2 drawn lighter
COUNTS = {"narrow": (2, 4, 8), "full": (4, 8, 16)}
SHADE = {2: 0.45, 4: 1.0, 8: 1.0, 16: 1.0}

style.setup()
fig, axes = plt.subplots(1, 2, figsize=(style.FULL, 2.3), sharey=True, constrained_layout=True)
centres = np.sqrt(SNR_EDGES[1:] * SNR_EDGES[:-1])
for ax, (band, counts) in zip(axes, COUNTS.items()):
    ax.axhspan(*FISHER_BAND, color=style.FILL, lw=0, zorder=0)
    ax.axhline(1.0, **style.GUIDE)
    for n in counts:
        rows = mcmc_reference.sources(band, n)
        snr = np.array([r["snr"] for r in rows])
        ratio = np.array([r["ratio"] for r in rows])
        f0, rest = [], []
        for lo, hi in zip(SNR_EDGES[:-1], SNR_EDGES[1:]):
            sel = (snr >= lo) & (snr < hi)
            ok = sel.sum() >= MIN_SOURCES
            f0.append(np.median(ratio[sel, 0]) if ok else np.nan)
            rest.append(np.median(np.median(ratio[sel][:, 2:], axis=1)) if ok else np.nan)
        ax.plot(centres, f0, color=COLORS[n], alpha=SHADE[n], lw=1.2, marker="o", ms=2.5, label=f"$S={n}$, $f_0$")
        ax.plot(centres, rest, color=COLORS[n], alpha=SHADE[n], lw=0.9, ls="--", label=f"$S={n}$, others")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(f"{band} band")
    ax.set_xlabel("source SNR")
    ax.set_xticks([2, 5, 10, 20, 50, 100, 200], ["2", "5", "10", "20", "50", "100", "200"])
    ax.set_xticks([], minor=True)
    ax.set_xlim(1.2, 400)
    ax.legend(loc="lower right", frameon=False, fontsize=6, ncol=2, handlelength=1.8, columnspacing=0.8)
axes[0].set_yticks([0.03, 0.1, 0.3, 1.0, 3.0], ["0.03", "0.1", "0.3", "1", "3"])
axes[0].set_yticks([], minor=True)
axes[0].set_ylabel("Fisher / MCMC width")
style.save(fig, "mcmc")
