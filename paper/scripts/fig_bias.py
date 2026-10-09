"""Figure: the f0 bias as a function of where the source sits in the window.

Median offset (flow median minus truth, in bins) of found sources at SNR >= 40, in 8 slices
of the source's f0 flow coordinate from -1 (low edge of the window) to +1. Shaded: about one
standard error of the median (1.25 sigma / sqrt(n)). At constant lr each run carries its own
smooth bias profile; the cooldown removes it (F19, F23).
"""

import numpy as np

import data
import style
from style import plt

style.setup()
coord = data.window_coordinate()
edges = np.linspace(-1, 1, 9)
centres = (edges[1:] + edges[:-1]) / 2
fig, ax = plt.subplots(figsize=(style.COLUMN, 1.7), constrained_layout=True)
for run in ["XS-1M", "XS-late-768", "XS-late-768-cool"]:
    sc = data.scorecard(run)
    keep = (sc["snr"] >= 40) & (sc["width"] < 1) & (sc["width"] > 0)
    med, err = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        o = sc["offset"][keep & (coord >= a) & (coord < b)]
        med.append(np.median(o))
        err.append(1.2533 * 1.4826 * np.median(np.abs(o - np.median(o))) / np.sqrt(o.size))
    med, err = np.array(med), np.array(err)
    meta = style.MODELS[run]
    ax.fill_between(centres, med - err, med + err, color=meta["color"], alpha=0.12, lw=0)
    ax.plot(centres, med, color=meta["color"], marker="o", ms=3.2, mec="white", mew=0.6, lw=1.2,
            label=meta["label"])
    print(f"{run}: max |median offset| {np.max(np.abs(med)):.3f} bins")
ax.axhline(0, color=style.AXIS, lw=0.7)
ax.set(xlabel=r"position of $f_0$ in the window (flow coordinate)", ylabel="median offset [bins]",
       xlim=(-1, 1))
ax.legend(loc="lower right", fontsize=5.6)
style.save(fig, "bias")
