"""Figure: training curves (per-epoch median flow loss, 1 epoch = 1000 steps).

(a) XS, uniform clock: 500k steps then continued to 1M at the same constant lr;
(b) XS, warped clock (p = 3, aux heads for the first 10%): 512 and 768 wide, each then cooled
    down over 200k more steps with the lr decaying linearly to 0;
(c) B1, the whole band: 200k steps in four 24 h jobs, the last 40k cooling down.
The warp re-weights the objective, so (a) and (b) are not on a common scale; (c) is a
different problem.
"""

import numpy as np

import data
import style
from style import plt

style.setup()
fig, axes = plt.subplots(1, 3, figsize=(style.FULL, 1.75), constrained_layout=True)


def line(ax, run, epochs=None, **kw):
    d = data.training_log(run)
    keep = np.ones_like(d["epoch"], bool) if epochs is None else (d["epoch"] >= epochs[0]) & (d["epoch"] <= epochs[1])
    meta = style.MODELS[run]
    ax.plot(d["epoch"][keep] / 1000, d["flow"][keep], color=meta["color"], lw=1.0, **kw)
    return d


# (a) uniform clock
ax = axes[0]
d = line(ax, "XS-1M")
off = d["epoch"][np.argmax(d["aux"] == 0)]
ax.annotate("aux heads off", (off / 1000, d["flow"][d["epoch"] == off][0]), xytext=(0.42, 0.53),
            fontsize=5.8, color=style.INK2, arrowprops=dict(arrowstyle="-", color=style.MUTED, lw=0.5))
ax.axvline(0.5, color=style.AXIS, lw=0.7)
ax.text(0.52, 0.66, "500k:\ncontinued", fontsize=5.8, color=style.INK2, va="top")
ax.set(xlabel="steps [M]", ylabel="flow loss", ylim=(0.39, 0.68), xlim=(0, 1.02))
ax.set_title("XS, uniform clock (512 wide)", fontsize=6.8, color=style.INK2, pad=3)
style.panel_label(ax, "a")

# (b) warped clock, 512 and 768, and their cooldowns
ax = axes[1]
ax.axvspan(1.0, 1.2, color=style.GRID, alpha=0.7, lw=0)
ax.text(1.1, 0.655, "cooldown\nlr → 0", ha="center", va="top", fontsize=5.8, color=style.INK2)
for run, rng, label in [("XS-late", (1, 1000), "512"), ("XS-late-768", (1, 1000), "768"),
                        ("XS-late-cool", (1001, 1200), "512, cooled"),
                        ("XS-late-768-cool", (1001, 1200), "768, cooled")]:
    d = data.training_log(run)
    keep = (d["epoch"] >= rng[0]) & (d["epoch"] <= rng[1])
    ax.plot(d["epoch"][keep] / 1000, d["flow"][keep], color=style.MODELS[run]["color"], lw=1.0, label=label)
ax.set(xlabel="steps [M]", ylim=(0.41, 0.68), xlim=(0, 1.22))
ax.set_title("XS, warped clock", fontsize=6.8, color=style.INK2, pad=3)
ax.legend(loc="upper right", bbox_to_anchor=(0.86, 1.0), fontsize=5.8, ncol=1)
style.panel_label(ax, "b")

# (c) B1
ax = axes[2]
d = data.training_log("B-late")
ax.axvspan(160, 200, color=style.GRID, alpha=0.7, lw=0)
ax.text(180, 0.765, "cooldown\nlr → 0", ha="center", va="top", fontsize=5.8, color=style.INK2)
ax.plot(d["epoch"], d["flow"], color=style.MODELS["B-late"]["color"], lw=1.0)
# the 24 h job boundaries: each restarts the optimizer's moments, with no visible cost
for start in (57, 113, 168):
    ax.axvline(start - 0.5, color=style.AXIS, lw=0.5, ls=(0, (2, 2)))
ax.annotate(f"{np.mean(d['flow'][-10:]):.3f}", (200, d["flow"][-1]), xytext=(128, 0.458),
            fontsize=5.8, color=style.INK2, arrowprops=dict(arrowstyle="-", color=style.MUTED, lw=0.5))
off = d["epoch"][np.argmax(d["aux"] == 0)]
ax.axvline(off, color=style.AXIS, lw=0.7)
ax.text(off + 3, 0.765, "aux\noff", fontsize=5.8, color=style.INK2, va="top")
ax.set(xlabel="steps [k]", ylim=(0.45, 0.78), xlim=(0, 205))
ax.set_title("B, warped clock (512 wide)", fontsize=6.8, color=style.INK2, pad=3)
style.panel_label(ax, "c")
style.save(fig, "training")
