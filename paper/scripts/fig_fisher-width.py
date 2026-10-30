"""Figure: the Fisher forecast's f0 width per eval source against the ideal sqrt(3)/(pi SNR), narrow and full band."""

import json
import math

import numpy as np

import style
from style import plt

FISHER = style.REPO / "reports/mcmc/data/fisher-diagonal.json"
SNR_BANDS = (15, 40, 100)

bands = json.loads(FISHER.read_text())["bands"]
# f0 at the start of the observation: its width with a free quadratic phase over the centred-origin one
gram = lambda n: np.linalg.inv(1 / (np.arange(n)[:, None] + np.arange(n) + 1))
start_factor = math.sqrt(gram(3)[1, 1] / gram(2)[1, 1])

style.setup()
fig, (ax_width, ax_ratio) = plt.subplots(
    1, 2, figsize=(style.FULL, 2.0), sharex=True, constrained_layout=True
)
for band, color, marker in [("narrow", style.BLUE, "o"), ("full", style.ORANGE, "D")]:
    injections = bands[band]["injections"]
    snr = np.concatenate([i["source_snr"] for i in injections])
    width = np.concatenate([i["f0_width_bins"] for i in injections])
    ideal = np.concatenate([i["ideal_width_bins"] for i in injections])
    for ax, y in [(ax_width, width), (ax_ratio, width / ideal)]:
        ax.plot(snr, y, ls="none", marker=marker, color=color, label=f"{band} band")

snr_grid = np.geomspace(6, 1200, 50)
ax_width.plot(snr_grid, math.sqrt(3) / (math.pi * snr_grid), **style.GUIDE)
style.direct_label(
    ax_width,
    snr_grid[8],
    math.sqrt(3) / (math.pi * snr_grid[8]),
    r"$\sqrt{3}/(\pi\rho)$",
    dx=-4,
    dy=-6,
)
ax_ratio.axhline(1, **style.GUIDE)
ax_ratio.axhline(start_factor, **style.GUIDE)
style.direct_label(
    ax_ratio,
    snr_grid[0],
    start_factor,
    r"$f_0$ at the start",
    dx=2,
    dy=5,
)
for ax in (ax_width, ax_ratio):
    for edge in SNR_BANDS:
        ax.axvline(edge, color=style.GRID, lw=0.8, zorder=0.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"source SNR $\rho$")
ax_ratio.set_yticks([1, 2, 4], ["1", "2", "4"])
ax_ratio.set_yticks([], minor=True)
ax_width.set_ylabel(r"Fisher $f_0$ width [bins]")
ax_ratio.set_ylabel("Fisher / ideal width")
ax_width.legend(loc="upper right")
style.panel_label(ax_width, "a")
style.panel_label(ax_ratio, "b")
style.save(fig, "fisher-width")
