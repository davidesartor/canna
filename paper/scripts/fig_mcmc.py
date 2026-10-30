"""Figure: Fisher widths against the joint reference MCMC, per source of every injection with chains."""

import json

import numpy as np

import style
from style import plt

DRAWS = style.REPO / "reports/mcmc/data/draws.json"
FISHER_BAND = (0.8, 1.25)

# the joint width is the mean over its seeds; the ratio is free of draws.json's scale
draws = json.loads(DRAWS.read_text())
drift = draws["columns"][1]
snr, fisher = [], []
for w in draws["windows"]:
    if not w["joint_width"]:
        continue
    for s, snr_s in enumerate(w["snr"]):
        joint_width = np.mean([seed[s] for seed in w["joint_width"]], axis=0)
        snr.append(snr_s)
        fisher.append(np.array(w["fisher_width"][s]) / joint_width)
snr, fisher = np.array(snr), np.array(fisher)

style.setup()
fig, ax = plt.subplots(figsize=(style.COLUMN, 2.2), constrained_layout=True)
others = [c for c in range(fisher.shape[1]) if c != 1]
ax.axhspan(*FISHER_BAND, color=style.FILL, lw=0, zorder=0)
ax.axhline(1.0, **style.GUIDE)
ax.vlines(
    snr,
    fisher[:, others].min(axis=1),
    fisher[:, others].max(axis=1),
    color=style.INK2,
    lw=0.8,
    zorder=2,
)
ax.scatter(
    snr,
    np.median(fisher[:, others], axis=1),
    s=9,
    color=style.BLUE,
    zorder=3,
    label="median of the other seven",
)
ax.scatter(
    snr,
    fisher[:, 1],
    s=11,
    marker="D",
    facecolor="none",
    edgecolor=style.ORANGE,
    lw=0.8,
    zorder=3,
    label={"log Mc": r"$\ln\mathcal{M}_c$", "log fdot": r"$\ln\dot f$"}[drift],
)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_yticks([0.3, 0.5, 1.0, 2.0, 3.0], ["0.3", "0.5", "1", "2", "3"])
ax.set_yticks([], minor=True)
ax.set_ylabel("Fisher / MCMC width")
ax.set_xlabel("source SNR")
ax.legend(loc="lower right", frameon=False, handletextpad=0.3)
style.save(fig, "mcmc")
