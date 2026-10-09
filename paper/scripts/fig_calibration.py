"""Figure: calibration of the f0 marginal on the 840 scorecard sources.

(a) the empirical CDF of the truth's rank among the flow draws (the fraction of draws below
    it), minus the uniform CDF, with a pointwise 95% band for n sources: a calibrated
    posterior stays inside, a biased one leans, an over-confident one makes an S.
(b) the fraction of truths inside the central 68% and 95% intervals, per SNR band.
"""

import numpy as np

import data
import style
from style import plt

style.setup()
RUNS = ["XS-late-768-cool", "XS-1M"]
fig, axes = plt.subplots(1, 2, figsize=(style.COLUMN, 1.6), constrained_layout=True)

ax = axes[0]
u = np.linspace(0, 1, 401)
n = len(data.scorecard(RUNS[0])["rank"])
band = 1.96 * np.sqrt(u * (1 - u) / n)
ax.fill_between(u, -band, band, color=style.GRID, alpha=0.8, lw=0, label="95% band")
for run in RUNS:
    r = np.sort(data.scorecard(run)["rank"])
    ecdf = np.searchsorted(r, u, side="right") / r.size
    meta = style.MODELS[run]
    ax.plot(u, ecdf - u, color=meta["color"], lw=1.1, label=meta["short"])
ax.axhline(0, color=style.AXIS, lw=0.7)
ax.set(xlabel="rank of the truth", ylabel="ECDF $-$ uniform", xlim=(0, 1))
ax.legend(loc="upper left", fontsize=5.6, handlelength=1.2)
style.panel_label(ax, "a")

ax = axes[1]
x = np.arange(4)
for k, run in enumerate(RUNS):
    sc = data.scorecard(run)
    meta = style.MODELS[run]
    for key, marker in (("in68", "o"), ("in95", "s")):
        cov = [np.mean(sc[key][(sc["snr"] >= a) & (sc["snr"] < b)]) for a, b in data.SNR_BANDS]
        ax.plot(x + (k - 0.5) * 0.18, cov, ls="", marker=marker, ms=3.2, color=meta["color"],
                mec="white", mew=0.5, label=f"{meta['short']}, {key[2:]}%")
for level in (0.68, 0.95):
    ax.axhline(level, color=style.INK, lw=0.6, ls=(0, (3, 2)))
ax.set_xticks(x, ["<15", "15-40", "40-100", "≥100"], fontsize=5.8)
ax.set(xlabel=r"source SNR $\rho$", ylabel="coverage", ylim=(0.6, 1.02))
leg = ax.legend(loc="lower right", fontsize=5.0, ncol=2, handletextpad=0.2, columnspacing=0.6,
                bbox_to_anchor=(1.0, 0.0), frameon=True, framealpha=1.0, borderpad=0.3)
leg.get_frame().set_facecolor("white")
leg.get_frame().set_edgecolor("white")
ax.set_ylim(0.52, 1.02)
style.panel_label(ax, "b")
style.save(fig, "calibration")
