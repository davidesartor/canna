"""Why each band's R is what it is, and what cutting a response to R bins costs."""

import json

import matplotlib.pyplot as plt
import numpy as np

import style
from canna.lisa import config
from canna.lisa.physics import (
    ANTENNA_HARMONICS,
    EARTH_ORBIT_SPEED,
    SPEED_OF_LIGHT,
    YEAR,
    chirp_mass_support,
    fdot_from_chirp_mass,
)
from canna.lisa.problem import LisaGB

REPORT = style.REPO / "reports/diagnostics/forward-model/data"
BANDS = {"narrow": style.BLUE, "full": style.ORANGE}

style.setup()
configs = {name: config.load(f"{name}-B").problem for name in BANDS}
problems = {name: LisaGB(**cfg) for name, cfg in configs.items()}
t_obs = problems["full"].t_obs
mc_hi = chirp_mass_support(problems["full"].mass_range)[1]
points = json.loads((REPORT / "response-points.json").read_text())
sources = np.load(REPORT / "response-points.npz")

fig, (ax_span, ax_offset, ax_ladder) = plt.subplots(
    1, 3, figsize=(style.FULL, 2.2), layout="constrained"
)

# (a) the heaviest binary's half-span and its three terms against each band's R/2
f = np.geomspace(1e-4, 2e-2, 400)
f_mhz = f * 1e3
doppler = f * EARTH_ORBIT_SPEED / SPEED_OF_LIGHT * t_obs
chirp = np.asarray(fdot_from_chirp_mass(mc_hi, f)) * t_obs**2
antenna = np.full_like(f, ANTENNA_HARMONICS * t_obs / YEAR)
for term, name, at, dy in (
    (doppler, "Doppler", 40, 5),
    (chirp, "chirp", 330, -8),
    (antenna, "antenna", 20, -5),
):
    ax_span.loglog(f_mhz, term, **style.GUIDE)
    style.direct_label(ax_span, f_mhz[at], term[at], name, dx=0, dy=dy, ha="left")
ax_span.loglog(f_mhz, doppler + chirp + antenna, color=style.INK)
style.direct_label(
    ax_span,
    f_mhz[300],
    (doppler + chirp + antenna)[300],
    "sum",
    dx=-3,
    dy=4,
    ha="right",
)
for name, color in BANDS.items():
    problem = problems[name]
    half = problem.response_points / 2
    holds = LisaGB(
        **(
            configs[name]
            | dict(f0_range=(1e-4, 1.0), response_points=problem.response_points)
        )
    ).f0_range[1]
    ax_span.axhline(half, color=color, lw=1.0)
    ax_span.plot([holds * 1e3], [half], color=color, marker="o", ls="none", zorder=3)
    style.direct_label(
        ax_span,
        0.11,
        half,
        rf"{name}: $R/2 = {half:.0f}$",
        dx=0,
        dy=4,
        ha="left",
        va="bottom",
    )
ax_span.set(
    xlabel=r"$f_0$ [mHz]", ylabel="half-span [bins]", xlim=(0.1, 20), ylim=(0.3, 3e3)
)
ax_span.set_title("Span of the heaviest binary")
style.panel_label(ax_span, "a")

# (b) each source's truncation error against its distance from a bin centre
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
    style.direct_label(
        ax_offset,
        0.5,
        np.median(sources[f"{name}_rel"][np.abs(sources[f"{name}_bin_offset"]) > 0.45]),
        name,
        dx=3,
    )
ax_offset.set(
    yscale="log",
    xlim=(0, 0.5),
    xlabel=r"$|f_0 T_{\rm obs} - q|$ [bins]",
    ylabel="relative error per source",
)
ax_offset.set_title("Leakage, not span")
style.panel_label(ax_offset, "b")

# (c) median relative error over a ladder of R, the band's own R ringed
for name, color in BANDS.items():
    ladder = np.asarray(points[f"{name}_ladder_points"])
    median = np.asarray(points[f"{name}_ladder_rel_median"])
    ax_ladder.loglog(ladder, median, color=color, marker="o", ms=2.6, mew=0, label=name)
    own = ladder == problems[name].response_points
    ax_ladder.plot(
        ladder[own],
        median[own],
        marker="o",
        ms=6,
        mfc="none",
        mec=color,
        ls="none",
    )
ax_ladder.legend(loc="lower left")
half_power = ladder.astype(float) ** -0.5
ax_ladder.loglog(ladder, half_power * median[-1] / half_power[-1], **style.GUIDE)
style.direct_label(
    ax_ladder,
    ladder[-3],
    (half_power * median[-1] / half_power[-1])[-3],
    r"$\propto R^{-1/2}$",
    dx=0,
    dy=-8,
    ha="center",
)
ax_ladder.set(
    xlabel=r"response points $R$",
    ylabel="median relative error",
    xlim=(ladder[0] / 2.2, ladder[-1] * 1.5),
)
ax_ladder.set_title(
    rf"Against $R = {points['reference_points']:,}$".replace(",", r"\,")
)
style.panel_label(ax_ladder, "c")

style.save(fig, "forward-model")
