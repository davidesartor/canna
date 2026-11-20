"""Figure: the source-count runs against the joint MCMC, and what a longer run buys (data/runs/source-count.json).

(a), (b) flow over joint-MCMC width of f0 and of the right ascension per source-SNR band, for each band and
source count's selected model (numbers_results.selected); (c) eval-catalogue NLL against the epoch at which the
cooldown starts, for the main run (at its planned 80% point) and its four cooldown forks.
"""

import numpy as np

import style
from style import plt

import numbers_results as nr  # noqa: E402  (rebuilds numbers/results.tex as a side effect)

# hue is the band, lightness the source count: darker with more sources
COUNTS = {"narrow": (2, 4, 8), "full": (4, 8, 16)}
SHADES = {
    "narrow": ("#f5b08f", style.ORANGE, "#9c3a12"),
    "full": (style.BLUE_RAMP[1], style.BLUE_RAMP[3], style.BLUE_RAMP[5]),
}
MODELS = [(band, n, color) for band in COUNTS for n, color in zip(COUNTS[band], SHADES[band])]
BANDS = ("0-7", "7-15", "15-40", ">=40")
BAND_LABELS = ("<7", "7–15", "15–40", "≥40")

style.setup()
fig, axes = plt.subplots(1, 3, figsize=(style.FULL, 2.3), layout="constrained")
x = np.arange(len(BANDS))

for ax, column, title in ((axes[0], "f0 [bins]", "Width of $f_0$"), (axes[1], "lon", r"Width of $\alpha$")):
    for band, n, color in MODELS:
        if nr.selected(band, n) is None:  # e64 not landed (full S=4)
            continue
        ev = nr.selected(band, n)["evals"][-1]
        y = [nr.row(ev, b, "vs_mcmc")["width_ratio"][column] if nr.row(ev, b, "vs_mcmc") else np.nan for b in BANDS]
        ax.plot(x, y, color=color, marker="o", label=f"{band}, $S = {n}$")
    ax.axhline(1.0, **style.GUIDE)
    ax.set_yscale("log")
    ax.set_yticks([1, 2, 5, 10, 20], ["1", "2", "5", "10", "20"])
    ax.set_yticks([], minor=True)
    ax.set_xticks(x, BAND_LABELS)
    ax.set_xlabel("source SNR")
    ax.set_title(title)
axes[0].set_ylabel("flow / MCMC width")
axes[1].sharey(axes[0])
axes[0].legend(ncol=2, loc="upper left")

ax = axes[2]
for band, n, color in MODELS:
    fam = {r["fork_epoch"]: r for r in nr.family(band, n)}
    if not fam:
        continue
    # a main run that stopped early (full-4 diverged before its cooldown) has no cooled point
    pts = [(e if e is not None else 0.8 * r["epochs_planned"], r["evals"][-1]["catalog_at_eval"])
           for e, r in fam.items() if e is not None or r["epochs_done"] == r["epochs_planned"]]
    pts.sort()
    ax.plot(*zip(*pts), color=color, marker="o")
ax.set_xscale("log", base=2)
ax.set_xticks([4, 8, 16, 32, 51.2], ["4", "8", "16", "32", "51 (main)"])
ax.minorticks_off()
ax.set_xlabel("cooldown starts at epoch")
ax.set_ylabel("eval-catalogue NLL")
ax.set_title("Longer runs")

for ax, letter in zip(axes, "abc"):
    style.panel_label(ax, letter)
style.save(fig, "results")
