"""Figure: what each change bought, on the same 840 scorecard sources.

(a) the loud-source floor: median f0 width of found sources at SNR >= 100, with its median
    ratio to the ideal width; (b) the found fraction (width < 1 bin) per SNR band.
Models in the order they were trained.
"""

import numpy as np

import data
import style
from style import plt

style.setup()
RUNS = ["XS-1M", "XS-late", "XS-late-768", "XS-late-cool", "XS-late-768-cool"]
fig, axes = plt.subplots(1, 2, figsize=(style.FULL, 1.75), constrained_layout=True,
                         gridspec_kw=dict(width_ratios=[1.0, 1.15]))

# (a) loud floor
ax = axes[0]
for i, run in enumerate(RUNS):
    sc = data.scorecard(run)
    loud = (sc["snr"] >= 100) & (sc["width"] < 1)
    w = np.median(sc["width"][loud])
    x_ideal = np.median(sc["width"][loud] / sc["ideal"][loud])
    meta = style.MODELS[run]
    ax.barh(i, w, height=0.55, color=meta["color"], lw=0)
    ax.text(w + 0.003, i, f"{w:.3f} bins, ×{x_ideal:.0f} ideal", va="center", fontsize=5.8, color=style.INK2)
ax.set_yticks(range(len(RUNS)), [style.MODELS[r]["label"] for r in RUNS], fontsize=6.2)
ax.invert_yaxis()
ax.set(xlabel=r"median $f_0$ width at $\rho\geq100$ [bins]", xlim=(0, 0.215))
ax.grid(axis="y", visible=False)
style.panel_label(ax, "a", x=-0.62)

# (b) detection per SNR band
ax = axes[1]
for run in RUNS:
    sc = data.scorecard(run)
    found = [np.mean(sc["width"][(sc["snr"] >= a) & (sc["snr"] < b)] < 1) for a, b in data.SNR_BANDS]
    meta = style.MODELS[run]
    ax.plot(range(4), found, color=meta["color"], marker="o", ms=3.2, mec="white", mew=0.6,
            lw=1.2, label=meta["label"])
counts = [np.sum((sc["snr"] >= a) & (sc["snr"] < b)) for a, b in data.SNR_BANDS]
ax.set_xticks(range(4), [f"{l}\n(n={n})" for l, n in zip(data.BAND_LABELS, counts)], fontsize=6.2)
ax.set(xlabel=r"source SNR $\rho$", ylabel="found fraction", ylim=(0.4, 1.02), xlim=(-0.2, 3.2))
ax.legend(loc="lower right", fontsize=5.8)
style.panel_label(ax, "b")
style.save(fig, "ladder")
