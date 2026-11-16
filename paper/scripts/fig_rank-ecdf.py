"""Figure: rank ECDF minus uniform of the validation pass (tab:results-validation), joint log-likelihood (top) and
overlap (bottom) per band and source count, the selected flow against its Gaussian mock, with the simultaneous 95%
band of the γ test for the flow's number of ranks. From data/runs/ranks.npz (import_ranks.py). A curve below zero
means ranks pile up high: the truth fits the data better than the draws."""

import numpy as np

import style
from style import plt

DATA = np.load(style.PAPER / "data/runs/ranks.npz")
CONFIGS = [("narrow", 2), ("narrow", 4), ("narrow", 8), ("full", 4), ("full", 8), ("full", 16)]
ROWS = [("loglik", "log-likelihood"), ("overlap", "overlap")]

style.setup()
fig, axes = plt.subplots(2, len(CONFIGS), figsize=(style.FULL, 2.9), sharex=True, sharey="row",
                         constrained_layout=True)
for col, (band, n) in enumerate(CONFIGS):
    for row, (q, label) in enumerate(ROWS):
        ax = axes[row, col]
        flow, mock = (f"{band}-{n}/{w}/{q}" for w in ("flow", "mock"))
        k = int(DATA[f"{flow}/k"])
        z = np.arange(1, k) / k
        lo, hi = DATA[f"{flow}/band"]
        ax.fill_between(z, lo, hi, color=style.FILL, lw=0, zorder=0)
        ax.axhline(0, **style.GUIDE)
        zm = np.arange(1, int(DATA[f"{mock}/k"])) / int(DATA[f"{mock}/k"])
        ax.plot(zm, DATA[f"{mock}/ecdf"], color=style.MUTED, ls="--", lw=1)
        ax.plot(z, DATA[f"{flow}/ecdf"], color=style.ORANGE, lw=1.3)
        if row == 0:
            ax.set_title(f"{band}, $S={n}$", fontsize=8)
        if col == 0:
            ax.set_ylabel(f"{label}\nECDF $-$ uniform")
        if row == 1:
            ax.set_xlabel("rank / draws")
            ax.set_xticks([0, 0.5, 1], ["0", "0.5", "1"])
fig.legend([plt.Line2D([], [], color=style.ORANGE, lw=1.3), plt.Line2D([], [], color=style.MUTED, ls="--", lw=1),
            plt.Rectangle((0, 0), 1, 1, color=style.FILL)],
           ["flow", "Gaussian mock", "95% simultaneous band (uniform ranks)"],
           loc="outside upper center", ncol=3)
style.save(fig, "rank-ecdf")
