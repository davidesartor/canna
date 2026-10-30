"""Why each band's R is what it is, and what cutting a response to R bins costs."""

import json
import math

import matplotlib.pyplot as plt
import numpy as np

import style
from canna.lisa import config
from canna.lisa.physics import SOURCE_INTERCEPT, SOURCE_SLOPE, response_span
from canna.lisa.problem import LisaGB

REPORT = style.REPO / "reports/diagnostics/forward-model/data"
SPAN = style.REPO / "reports/prior/response-span/data"
BANDS = {"narrow": style.ORANGE, "full": style.BLUE}
F0_BINS = 20
SHARES = {0.5: style.BLUE_RAMP[2], 0.99: style.BLUE_RAMP[5]}


def quantile(a: np.ndarray, w: np.ndarray, y: float) -> float:
    """The smallest value with at least a share y of the weight w at or below it."""
    order = np.argsort(a, kind="stable")
    cumulative = np.cumsum(w[order])
    at = np.searchsorted(cumulative, y * cumulative[-1] * (1 - 1e-12))
    return float(a[order][min(at, a.size - 1)])


style.setup()
configs = {name: config.load(f"{name}-B").problem for name in BANDS}
problems = {name: LisaGB(**cfg) for name, cfg in configs.items()}
t_obs = problems["full"].t_obs
points = json.loads((REPORT / "response-points.json").read_text())
sources = np.load(REPORT / "response-points.npz")
span = json.loads((SPAN / "response-span.json").read_text())
widths = np.load(SPAN / "response-span.npz")

fig, (ax_span, ax_offset, ax_ladder) = plt.subplots(
    1, 3, figsize=(style.FULL, 2.2), layout="constrained"
)

# (a) the bins holding all of a source's above-noise power, per f0 bin, against the span at the mean fdot and each band's R
draws = span["amplitudes"]
w = np.repeat(widths["f0_weight"] * widths["seen"] / draws, draws)
width = widths["width"].reshape(w.size, -1)[:, span["keep"].index(1.0)]
edges = np.linspace(*np.log10(span["f0_range"]), F0_BINS + 1)
bins = np.repeat(
    np.clip(np.digitize(np.log10(widths["f0"]), edges) - 1, 0, F0_BINS - 1), draws
)
centres = 10 ** ((edges[:-1] + edges[1:]) / 2) * 1e3
for y, color in SHARES.items():
    # a y-quantile only where at least 1/(1 - y) effective sources lie above the noise
    q = []
    for b in range(F0_BINS):
        wb = w[bins == b]
        enough = wb.sum() > 0 and wb.sum() ** 2 / (wb**2).sum() * (1 - y) >= draws
        q.append(quantile(width[bins == b], wb, y) if enough else np.nan)
    ax_span.loglog(
        centres, q, color=color, marker="o", ms=2.4, mew=0, label=f"{y:.0%} of sources"
    )
f = np.geomspace(1e-4, problems["full"].f0_range[1], 200)
hard = [
    response_span(
        x, 10 ** (SOURCE_INTERCEPT[0] + SOURCE_SLOPE[0] * math.log10(x)), t_obs
    )
    for x in f
]
ax_span.loglog(
    f * 1e3, hard, color=style.INK, lw=1.0, label=r"$n_{\rm span}$, mean $\dot f$"
)
for name, color in BANDS.items():
    problem = problems[name]
    top = problem.f0_range[1] * 1e3
    ax_span.plot([0.1, top], [problem.response_points] * 2, color=color, lw=1.0)
    style.direct_label(
        ax_span,
        0.11,
        problem.response_points,
        rf"{name}: $R = {problem.response_points}$",
        dx=0,
        dy=3,
        ha="left",
        va="bottom",
    )
ax_span.legend(loc="lower right")
ax_span.set(
    xlabel=r"$f_0$ [mHz]",
    ylabel="bins above the noise",
    xlim=(0.1, 12.5),
    ylim=(2, 600),
)
ax_span.set_title("Span against $R$")
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
