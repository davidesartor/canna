"""The simulator and likelihood against Strub et al. (2022)'s single-source posterior."""

import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

import style

DATA = style.REPO / "reports/diagnostics/strub-source/data"
PULLS = {
    "f0": r"$f_0$",
    "log10_fdot": r"$\log_{10}\dot f$",
    "log10_A": r"$\log_{10}\mathcal{A}$",
    "ecl_lon": r"$\lambda$",
    "ecl_lat": r"$\beta$",
    "psi": r"$\psi$",
    "cos_iota": r"$\cos\iota$",
    "phi0": r"$\phi_0$",
}
CENTRED = 0.25
PRIOR, BOX, STRUB, NOISELESS = style.BLUE, style.AQUA, style.ORANGE, style.VIOLET


def density(ax, samples: np.ndarray, bins: np.ndarray, color: str, label: str) -> None:
    """Normalised histogram as a step line over a light wash."""
    heights, _ = np.histogram(samples, bins, density=True)
    ax.stairs(heights, bins, color=color, alpha=0.15, fill=True, lw=0)
    ax.stairs(heights, bins, color=color, lw=1.0, label=label)


def intervals(ax, rows: list[tuple[str, float, float, float, str]]) -> None:
    """68% intervals, one labelled row each, top to bottom."""
    for y, (_, lo, mid, hi, color) in enumerate(reversed(rows)):
        ax.plot([lo, hi], [y, y], color=color, lw=1.6)
        ax.plot([mid], [y], color=color, marker="o", zorder=3)
    ax.set(
        ylim=(-0.6, len(rows) - 0.4),
        yticks=range(len(rows)),
        yticklabels=[row[0] for row in reversed(rows)],
    )
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)


def density_axis(ax) -> None:
    ax.set(yticks=[])
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="x", labelbottom=False)


style.setup()
record = json.loads((DATA / "strub_source.json").read_text())
seed0 = np.load(DATA / "seed0.npz")

fig = plt.figure(figsize=(style.FULL, 3.9), layout="constrained")
top, bottom = fig.subfigures(2, 1, height_ratios=[1, 1.2])
ax_pull, ax_gain = top.subplots(1, 2)
f0_cell, cos_cell = bottom.subfigures(1, 2)
ax_f0, ax_f0_ci = f0_cell.subplots(2, 1, sharex=True, height_ratios=[2.2, 1])
ax_cos, ax_cos_ci = cos_cell.subplots(2, 1, sharex=True, height_ratios=[2.2, 1])

# (a) noiseless pulls per marginal
pulls = np.array([record[f"noiseless_pull_{name}"] for name in PULLS])
rows = np.arange(len(PULLS))[::-1]
ax_pull.axvspan(-CENTRED, CENTRED, color=style.FILL, lw=0, zorder=0)
ax_pull.axvline(0, **style.GUIDE)
ax_pull.hlines(rows, 0, pulls, color=NOISELESS, lw=1.2)
ax_pull.plot(pulls, rows, color=NOISELESS, marker="o", ls="none", zorder=3)
ax_pull.set(
    yticks=rows,
    yticklabels=list(PULLS.values()),
    xlim=(-1.6, 1.6),
    ylim=(-0.6, len(PULLS) - 0.4),
    xlabel="(posterior median − truth) / posterior std",
)
ax_pull.tick_params(axis="y", length=0)
ax_pull.grid(axis="y", visible=False)
ax_pull.spines["left"].set_visible(False)
ax_pull.set_title("Noiseless data: pulls")
style.panel_label(ax_pull, "a")

# (b) SNR gain from fitting over noise realisations, as a dot histogram
gain = np.array(record["sweep_rho_mle_list"]) - np.array(record["sweep_rho_obs_list"])
width = 0.05
column = np.floor(gain / width).astype(int)
stack_height = np.zeros_like(column)
filled: dict[int, int] = {}
for i in np.argsort(gain):
    filled[column[i]] = filled.get(column[i], 0) + 1
    stack_height[i] = filled[column[i]]
ax_gain.plot(
    (column + 0.5) * width, stack_height, color=PRIOR, marker="o", ls="none", zorder=3
)
mean, sem = record["sweep_gain_obs_mean"], record["sweep_gain_obs_sem"]
bar_y = stack_height.max() + 1.6
ax_gain.plot([mean - sem, mean + sem], [bar_y, bar_y], color=style.INK, lw=2.4)
ax_gain.plot([mean], [bar_y], color=style.INK, marker="o", zorder=3)
style.direct_label(ax_gain, mean + sem, bar_y, "mean ± s.e.", dx=4)
ax_gain.axvline(record["strub_gain"], color=STRUB, lw=1.2, zorder=1)
style.direct_label(
    ax_gain, record["strub_gain"], bar_y + 1.4, "Strub et al.", dx=-3, ha="right"
)
ax_gain.set(
    xlabel=r"$\rho_{\rm MLE} - \rho_{\rm obs}$",
    ylabel="noise realisations",
    ylim=(0, bar_y + 2.4),
)
ax_gain.yaxis.set_major_locator(MaxNLocator(integer=True))
ax_gain.set_title("SNR gain from fitting the noise")
style.panel_label(ax_gain, "b")

# (c) f0 posterior widths: Strub's prior, the Fisher box, and Strub et al.'s interval
bins = np.linspace(-5, 5, 81)
density(ax_f0, seed0["strub_f0_nhz"], bins, PRIOR, "Strub's prior")
density(ax_f0, seed0["box_f0_nhz"], bins, BOX, "Fisher box")
ax_f0.legend(loc="upper left")
density_axis(ax_f0)
ax_f0.set_title(r"Seed-0 posterior of $f_0$")
style.panel_label(ax_f0, "c", x=-0.04)
intervals(
    ax_f0_ci,
    [
        (
            "Strub et al.",
            -record["strub_f0_minus_nhz"],
            0,
            record["strub_f0_plus_nhz"],
            STRUB,
        ),
        (
            "Strub's prior",
            -record["seed0_f0_minus_nhz"],
            0,
            record["seed0_f0_plus_nhz"],
            PRIOR,
        ),
        ("Fisher box", -record["box_f0_minus_nhz"], 0, record["box_f0_plus_nhz"], BOX),
    ],
)
ax_f0_ci.set(xlim=(-5, 5), xlabel=r"$f_0$ − median [nHz]")

# (d) cos iota: the Fisher box cuts the face-on tail
bins = np.linspace(0, 1, 41)
density(ax_cos, seed0["strub_cos_iota"], bins, PRIOR, "Strub's prior")
density(ax_cos, seed0["box_cos_iota"], bins, BOX, "Fisher box")
truth = np.cos(record["iota"])
box_edge = record["seed0_ml_cos_iota"] + record["box_half_width_sin_iota"]
ax_cos.axvline(truth, color=style.INK, lw=0.8)
ax_cos.axvline(box_edge, **style.GUIDE)
top_y = ax_cos.get_ylim()[1] * 1.15
ax_cos.set_ylim(top=top_y)
style.direct_label(ax_cos, truth, top_y, "truth", dx=3, va="top")
style.direct_label(ax_cos, box_edge, top_y, "box edge", dx=3, va="top")
density_axis(ax_cos)
ax_cos.set_title(r"Seed-0 posterior of $\cos\iota$")
style.panel_label(ax_cos, "d", x=-0.04)
iota, plus, minus = (
    record["strub_iota_median"],
    record["strub_iota_plus"],
    record["strub_iota_minus"],
)
intervals(
    ax_cos_ci,
    [
        (
            "Strub et al.",
            np.cos(iota + plus),
            np.cos(iota),
            np.cos(iota - minus),
            STRUB,
        ),
        (
            "Strub's prior",
            record["seed0_cos_iota_q16"],
            record["seed0_median_cos_iota"],
            record["seed0_cos_iota_q84"],
            PRIOR,
        ),
        (
            "Fisher box",
            record["box_cos_iota_q16"],
            record["box_median_cos_iota"],
            record["box_cos_iota_q84"],
            BOX,
        ),
    ],
)
ax_cos_ci.set(xlim=(0, 1), xlabel=r"$\cos\iota$")

style.save(fig, "strub-source")
