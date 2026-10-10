"""Figure: B's f0 posterior widths, against XS's, in bins and in the flow coordinate.

(a) f0 width against SNR in bins of 1/T_obs: every B source (violet) and the final XS
    model's (blue), with the ideal width. B finds almost nothing.
(b) The same widths in units of the f0 flow coordinate x_f, which maps the window interior
    onto [-1, 1]: one unit is 24 bins on XS and ~1536 on B. The loud floors meet.
(c) B's widths against the bfloat16 cell of the input coordinate x_f, in bins: they do not
    follow it. Grey line: width = cell.
"""

import numpy as np

import data
import style
from style import plt

style.setup()
fig, axes = plt.subplots(1, 3, figsize=(style.FULL, 2.1), constrained_layout=True)
b, xs = data.scorecard("B-late"), data.scorecard("XS-late-768-cool")
coord = data.b_coordinate()
kept = b["width"] < 50
B, XS = style.MODELS["B-late"], style.MODELS["XS-late-768-cool"]
rho = np.logspace(np.log10(5), np.log10(2000), 50)
point = dict(s=4, lw=0, alpha=0.55, zorder=3)

# (a) in bins
ax = axes[0]
ax.scatter(xs["snr"], xs["width"], color=XS["color"], label="XS final", **dict(point, alpha=0.25))
ax.scatter(b["snr"], b["width"], color=B["color"], label="B", **point)
ax.plot(rho, np.sqrt(3) / (np.pi * rho), color=style.INK, lw=1.0, ls=(0, (4, 2)), zorder=4,
        label=r"ideal, $\sqrt{3}/(\pi\rho)$")
ax.axhline(1.0, color=style.MUTED, lw=0.7)
ax.text(1900, 1.25, "found: < 1 bin", fontsize=5.8, color=style.INK2, ha="right")
ax.set(xscale="log", yscale="log", xlim=(5, 2000), ylim=(3e-4, 3e3), xlabel=r"source SNR $\rho$",
       ylabel=r"$f_0$ width [bins]")
ax.legend(loc="upper right", fontsize=5.8)
style.panel_label(ax, "a")

# (b) in units of x_f
ax = axes[1]
ax.scatter(xs["snr"], xs["width"] / data.XS_BINS_PER_UNIT, color=XS["color"], **dict(point, alpha=0.25))
ax.scatter(b["snr"], b["width"] / coord["bins_per_x"], color=B["color"], **point)
for sc, per_unit, meta in [(xs, data.XS_BINS_PER_UNIT, XS), (b, coord["bins_per_x"], B)]:
    found = sc["width"] < (1 if sc is xs else 50)
    loud = found & (sc["snr"] >= 100)
    level = np.median((sc["width"] / per_unit)[loud])
    name = "B" if sc is b else "XS final"
    ax.axhline(level, color=meta["color"], lw=0.9, ls=(0, (3, 1.5)), zorder=4,
               label=rf"{name}, loud median {1e3 * level:.1f}$\times10^{{-3}}$")
ax.legend(loc="lower left", fontsize=5.8, title=r"$\rho\geq100$", title_fontsize=5.8)
ax.set(xscale="log", yscale="log", xlim=(5, 2000), ylim=(1e-5, 2), xlabel=r"source SNR $\rho$",
       ylabel=r"$f_0$ width [units of $x_f$]")
style.panel_label(ax, "b")

# (c) against the bf16 cell of the input coordinate
ax = axes[2]
loud = kept & (b["snr"] >= 40)
ax.scatter(coord["cell"][kept & ~loud], b["width"][kept & ~loud], color=B["color"], label=r"B, $\rho<40$",
           **dict(point, alpha=0.3))
ax.scatter(coord["cell"][loud], b["width"][loud], color=B["color"], marker="^", label=r"B, $\rho\geq40$",
           **dict(point, s=6))
cells = np.logspace(-3, 1, 10)
ax.plot(cells, cells, color=style.MUTED, lw=0.8, label="width = cell")
ax.set(xscale="log", yscale="log", xlim=(1e-3, 10), ylim=(0.3, 30),
       xlabel=r"bfloat16 cell of the input $x_f$ [bins]", ylabel=r"$f_0$ width [bins]")
ax.legend(loc="upper left", fontsize=5.8)
style.panel_label(ax, "c")
style.save(fig, "b_results")
