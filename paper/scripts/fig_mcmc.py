"""Figure: Fisher and narrowband MCMC widths against the joint MCMC, per source of the four joint windows."""

import json
import re

import numpy as np

import style
from style import plt

JOINT = style.REPO / "reports/mcmc/data"
COMPARE = style.REPO / "reports/posterior-eval/flow-vs-reference/data"
FISHER_BAND = (0.8, 1.25)

run = next(JOINT.glob("joint-*"))
pooled = {
    (c["inj"], c["src"]): np.array(c["width"])
    for c in json.loads((run / "chains.json").read_text())["narrowband_pooled"]
}

# one compare file per window: the MCMC and Fisher rows are the same for every flow run
compares = {
    int(re.search(r"window(\d+)", p.name)[1]): p
    for p in sorted((COMPARE / run.name).glob("compare_window*.npz"))
}
snr, fisher, narrow = [], [], []
for j, path in sorted(compares.items()):
    c = np.load(path)
    rows = list(c["rows"])
    for s in range(len(c["snr"])):
        joint_width = c[f"s{s}"][rows.index("mcmc width")]
        snr.append(float(c["snr"][s]))
        fisher.append(c[f"s{s}"][rows.index("fisher/mcmc width")])
        narrow.append(pooled[(j, s)] / joint_width)
snr, fisher, narrow = np.array(snr), np.array(fisher), np.array(narrow)

style.setup()
fig, axes = plt.subplots(
    1, 2, figsize=(style.FULL, 2.2), sharex=True, constrained_layout=True
)
others = [c for c in range(fisher.shape[1]) if c != 1]
for ax, ratio, label, letter in [
    (axes[0], fisher, "Fisher / joint MCMC width", "a"),
    (axes[1], narrow, "narrowband / joint MCMC width", "b"),
]:
    ax.axhspan(*FISHER_BAND, color=style.FILL, lw=0, zorder=0)
    ax.axhline(1.0, **style.GUIDE)
    ax.vlines(
        snr,
        ratio[:, others].min(axis=1),
        ratio[:, others].max(axis=1),
        color=style.INK2,
        lw=0.8,
        zorder=2,
    )
    ax.scatter(
        snr,
        np.median(ratio[:, others], axis=1),
        s=9,
        color=style.BLUE,
        zorder=3,
        label="median of the other seven",
    )
    ax.scatter(
        snr,
        ratio[:, 1],
        s=11,
        marker="D",
        facecolor="none",
        edgecolor=style.ORANGE,
        lw=0.8,
        zorder=3,
        label=r"$\ln\mathcal{M}_c$",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_yticks([0.3, 0.5, 1.0, 2.0, 3.0], ["0.3", "0.5", "1", "2", "3"])
    ax.set_yticks([], minor=True)
    ax.set_ylabel(label)
    style.panel_label(ax, letter)
axes[0].legend(loc="lower right", frameon=False, handletextpad=0.3)
ylim = (min(ax.get_ylim()[0] for ax in axes), max(ax.get_ylim()[1] for ax in axes))
for ax in axes:
    ax.set_ylim(ylim)
    ax.set_xlabel("source SNR")
style.save(fig, "mcmc")
