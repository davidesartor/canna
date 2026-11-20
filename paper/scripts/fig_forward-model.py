"""What cutting a response to R bins costs: leakage between bins, falling with R."""

import json

import matplotlib.pyplot as plt
import numpy as np

import style
from canna.lisa import config
from canna.lisa.problem import LisaGB

REPORT = style.REPO / "reports/diagnostics/forward-model/data"
BANDS = {"narrow": style.ORANGE, "full": style.BLUE}


style.setup()
configs = {name: config.load(f"{name}-B").problem for name in BANDS}
problems = {name: LisaGB(**cfg) for name, cfg in configs.items()}
points = json.loads((REPORT / "response-points.json").read_text())
sources = np.load(REPORT / "response-points.npz")

fig, (ax_offset, ax_ladder) = plt.subplots(
    1, 2, figsize=(0.7 * style.FULL, 2.2), layout="constrained"
)

# (a) each source's truncation error against its distance from a bin centre
for name, color in BANDS.items():
    ax_offset.plot(
        np.abs(sources[f"{name}_bin_offset"]),
        sources[f"{name}_rel"],
        color=color,
        marker="o",
        ms=1.6,
        mew=0,
        alpha=0.35,
        ls="none",
        rasterized=True,
    )
    ax_offset.plot([], [], color=color, marker="o", ms=3, ls="none", label=f"{name} band")
ax_offset.set(
    yscale="log",
    xlim=(0, 0.5),
    xlabel=r"$|f_0 T_{\rm obs} - q|$ [bins]",
    ylabel="relative error per source",
)
ax_offset.legend(loc="lower right")
ax_offset.set_title("Error against bin offset")
style.panel_label(ax_offset, "a")

# (b) median relative error over a ladder of R, the band's own R ringed
for name, color in BANDS.items():
    ladder = np.asarray(points[f"{name}_ladder_points"])
    median = np.asarray(points[f"{name}_ladder_rel_median"])
    ax_ladder.loglog(ladder, median, color=color, marker="o", ms=3)
    own = ladder == problems[name].response_points
    ax_ladder.plot(
        ladder[own],
        median[own],
        marker="o",
        ms=7,
        mfc="none",
        mec=color,
        mew=0.8,
        ls="none",
    )
half_power = ladder.astype(float) ** -0.5
ax_ladder.loglog(ladder, half_power * median[-1] / half_power[-1], **style.GUIDE)
style.direct_label(
    ax_ladder,
    ladder[2],
    (half_power * median[-1] / half_power[-1])[2],
    r"$\propto R^{-1/2}$",
    dx=-2,
    dy=-7,
    ha="right",
)
ax_ladder.plot([], [], marker="o", ms=7, mfc="none", mec=style.INK2, mew=0.8, ls="none", label="the band's own $R$")
ax_ladder.legend(loc="upper right")
ax_ladder.set(
    xlabel=r"response points $R$",
    ylabel="median relative error",
    xlim=(ladder[0] / 2.2, ladder[-1] * 1.5),
)
ax_ladder.set_title("Error against $R$")
style.panel_label(ax_ladder, "b")

style.save(fig, "forward-model")
