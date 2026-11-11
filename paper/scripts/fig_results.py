"""Figure: the source-count runs against the joint MCMC, and what a longer run buys (data/runs/source-count.json).

(a), (b) flow over joint-MCMC width of f0 and of the right ascension per source-SNR band, for each band and
source count's selected model (numbers_results.selected); (c) eval-catalogue NLL against the epoch at which the
cooldown starts, for the main run (at its planned 80% point) and its four cooldown forks.
"""

import numpy as np

import style
from style import plt

import numbers_results as nr  # noqa: E402  (rebuilds numbers/results.tex as a side effect)

CONFIG_STYLE = {
    ("narrow", 2): (style.BLUE, "-", "narrow, 2"),
    ("narrow", 4): (style.AQUA, "-", "narrow, 4"),
    ("narrow", 8): (style.VIOLET, "-", "narrow, 8"),
    ("full", 4): (style.ORANGE, (0, (4, 2)), "full, 4"),
    ("full", 8): (style.RED, (0, (4, 2)), "full, 8"),
    ("full", 16): (style.YELLOW, (0, (4, 2)), "full, 16"),
}
BANDS = ("0-7", "7-15", "15-40", ">=40")
BAND_LABELS = ("<7", "7–15", "15–40", "≥40")

style.setup()
fig, axes = plt.subplots(1, 3, figsize=(style.FULL, 2.3), layout="constrained")
x = np.arange(len(BANDS))

for ax, column, title in ((axes[0], "f0 [bins]", "$f_0$"), (axes[1], "lon", r"$\alpha$")):
    for (band, n), (color, ls, name) in CONFIG_STYLE.items():
        if not nr.family(band, n):
            continue
        ev = nr.selected(band, n)["evals"][-1]
        y = [nr.row(ev, b, "vs_mcmc")["width_ratio"][column] if nr.row(ev, b, "vs_mcmc") else np.nan for b in BANDS]
        ax.plot(x, y, color=color, ls=ls, marker="o", ms=2.8, mec="white", mew=0.5, lw=1.0, label=name)
    ax.axhline(1.0, **style.GUIDE)
    ax.set_yscale("log")
    ax.set_xticks(x, BAND_LABELS)
    ax.set_xlabel("source SNR")
    ax.set_title(f"flow / MCMC width, {title}", loc="left")
axes[0].set_ylabel("width ratio")
axes[1].sharey(axes[0])
axes[0].legend(frameon=False, fontsize=6, ncol=2, loc="upper left")

ax = axes[2]
for (band, n), (color, ls, name) in CONFIG_STYLE.items():
    fam = {r["fork_epoch"]: r for r in nr.family(band, n)}
    if not fam:
        continue
    # a main run that stopped early (full-4 diverged before its cooldown) has no cooled point
    pts = [(e if e is not None else 0.8 * r["epochs_planned"], r["evals"][-1]["catalog_at_eval"])
           for e, r in fam.items() if e is not None or r["epochs_done"] == r["epochs_planned"]]
    pts.sort()
    ax.plot(*zip(*pts), color=color, ls=ls, marker="o", ms=2.8, mec="white", mew=0.5, lw=1.0)
ax.set_xscale("log", base=2)
ax.set_xticks([4, 8, 16, 32, 51.2], ["4", "8", "16", "32", "51 (main)"])
ax.minorticks_off()
ax.set_xlabel("cooldown starts at epoch")
ax.set_ylabel("eval-catalogue NLL")
ax.set_title("longer runs", loc="left")

for ax, letter in zip(axes, "abc"):
    style.panel_label(ax, letter)
style.save(fig, "results")
