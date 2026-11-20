"""The band divided where the response width doubles, and where the visible sources fall (the response-span report)."""

import json
import math

import matplotlib.pyplot as plt
import numpy as np
import yaml

import style

SPAN = style.REPO / "reports/prior/response-span/data"
BANDS = style.REPO / "src/canna/lisa/configs/band"
F0_BINS = 40
BAND_R = (16, 32, 64, 128, 256)
BAND_NAMES = ("low tail", "low freq", "core", "high freq", "high tail")
GUIDE_R = tuple(2**k for k in range(3, 10))
VISIBLE_SNR = 7
KEEP, SHARE = 1.0, 0.999
# the (log10 f0, log10 width) density, as on the report's page: cells per axis, one-cell smoothing, contour masses
CELLS = 64
SMOOTH = np.array([0.054, 0.242, 0.399, 0.242, 0.054])
MASSES = (0.5, 0.9, 0.99)
WIDTH_RANGE = (1.0, 2048.0)


def pow2(q: float) -> int:
    """The smallest power of two holding q bins."""
    return 1 << max(1, math.ceil(math.log2(max(q, 2))))


def quantile(a: np.ndarray, w: np.ndarray, y: float) -> float:
    """The smallest value with at least a share y of the weight w at or below it."""
    order = np.argsort(a, kind="stable")
    cumulative = np.cumsum(w[order])
    at = np.searchsorted(cumulative, y * cumulative[-1] * (1 - 1e-12))
    return float(a[order][min(at, a.size - 1)])


style.setup()
span = json.loads((SPAN / "response-span.json").read_text())
z = np.load(SPAN / "response-span.npz")
draws = span["amplitudes"]
f0 = np.repeat(z["f0"], draws)
w = np.repeat(z["f0_weight"] * z["seen"] / draws, draws)
snr = 10 ** z["log_a"].astype(np.float64).ravel() * np.repeat(z["snr_unit"], draws)
width = z["width"].reshape(w.size, -1)[:, span["keep"].index(KEEP)].astype(float)
log_range = np.log10(span["f0_range"])

# R per log-f0 bin and the band edges where it first exceeds each band's R, as numbers_narrow-band.py
edges = np.linspace(*log_range, F0_BINS + 1)
bins = np.clip(np.digitize(np.log10(f0), edges) - 1, 0, F0_BINS - 1)
need = np.array(
    [pow2(quantile(width[bins == b], w[bins == b], SHARE)) for b in range(F0_BINS)]
)
first_over = [int(np.argmax(need > r)) for r in BAND_R[:-1]]
band_edges = [
    10 ** log_range[0],
    *(10 ** edges[b] for b in first_over),
    10 ** log_range[1],
]
band_mhz = np.array(band_edges) * 1e3

fig, (ax_width, ax_share) = plt.subplots(
    2,
    1,
    figsize=(style.COLUMN, 3.9),
    sharex=True,
    height_ratios=(1.5, 1),
    layout="constrained",
)
for ax in (ax_width, ax_share):
    for lo, hi, shade in zip(band_mhz, band_mhz[1:], (True, False, True, False, True)):
        if shade:
            ax.axvspan(lo, hi, color=style.FILL, lw=0, zorder=0)

# (a) density of the sources above the noise in (f0, bins holding all their above-noise power)
jitter = np.random.default_rng(0).uniform(-1.0, 1.0, width.size)
log_width = np.log10(np.clip(width + jitter, *WIDTH_RANGE))
rx, ry = (log_range[0], log_range[1]), np.log10(WIDTH_RANGE)
h = np.histogram2d(np.log10(f0), log_width, CELLS, [rx, ry], weights=w)[0].T
for axis in (0, 1):
    h = np.apply_along_axis(np.convolve, axis, h, SMOOTH, mode="same")
ordered = np.sort(h.ravel())[::-1]
held = np.cumsum(ordered) / ordered.sum()
levels = sorted(ordered[np.searchsorted(held, m)] for m in MASSES)
x_cells = 10 ** np.linspace(*rx, CELLS) * 1e3
y_cells = 10 ** np.linspace(*ry, CELLS)
ax_width.contourf(
    x_cells,
    y_cells,
    h,
    levels=[*levels, ordered[0] * 1.01],
    colors=style.BLUE_RAMP[0:5:2],
    zorder=1,
)
ax_width.contour(
    x_cells,
    y_cells,
    h,
    levels=levels,
    colors=style.BLUE_RAMP[4],
    linewidths=0.5,
    zorder=1.2,
)

# every power-of-two R as a guide, each band's R in ink, and the top band's per-f0 need
for r in GUIDE_R:
    ax_width.axhline(r, color=style.MUTED, lw=0.5, ls=(0, (2, 2)), zorder=1.5)
    style.direct_label(ax_width, band_mhz[0], r, rf"$R={r}$", dx=2, dy=1, ha="left", va="bottom", fontsize=5.5)
for r, lo, hi in zip(BAND_R, band_mhz, band_mhz[1:]):
    ax_width.plot([lo, hi], [r, r], color=style.INK, lw=1.0, zorder=2)
ax_width.stairs(
    need[first_over[-1] :],
    10 ** edges[first_over[-1] :] * 1e3,
    color=style.INK,
    lw=1.0,
    ls=(0, (2, 1.5)),
    baseline=None,
    zorder=2,
)
ax_width.set(
    xscale="log",
    yscale="log",
    ylim=(2, 2600),
    ylabel="bins above the noise",
)
ax_width.set_title("Response width against $f_0$")
style.panel_label(ax_width, "a")

# (b) share of the catalogue and of the visible sources per dex of f0, with each band's visible share
dex = (log_range[1] - log_range[0]) / F0_BINS
catalogue = np.histogram(np.log10(z["f0"]), edges, weights=z["f0_weight"])[0]
visible_w = w * (snr >= VISIBLE_SNR)
visible = np.histogram(np.log10(f0), edges, weights=visible_w)[0]
centres = 10**edges * 1e3
ax_share.stairs(catalogue / catalogue.sum() / dex, centres, color=style.MUTED, lw=1.0)
ax_share.stairs(
    visible / visible.sum() / dex,
    centres,
    color=style.BLUE,
    fill=True,
    alpha=0.55,
    lw=0,
)
ax_share.stairs(visible / visible.sum() / dex, centres, color=style.BLUE, lw=1.0)
style.direct_label(
    ax_share,
    0.2,
    1.05 * (catalogue / catalogue.sum() / dex).max(),
    "catalogue",
    dx=0,
    dy=2,
    ha="center",
    va="bottom",
)
peak = int(np.argmax(visible))
style.direct_label(
    ax_share,
    centres[peak],
    0.8 * (visible / visible.sum() / dex)[peak],
    rf"SNR $\geq$ {VISIBLE_SNR}",
    dx=-6,
)
for name, lo, hi in zip(BAND_NAMES, band_mhz, band_mhz[1:]):
    inside = (f0 >= lo / 1e3) & (f0 < hi / 1e3)
    ax_share.annotate(
        f"{name}\n{100 * visible_w[inside].sum() / visible_w.sum():.2g}%",
        (math.sqrt(lo * hi), 0.97),
        xycoords=("data", "axes fraction"),
        ha="center",
        va="top",
        fontsize=5.8,
        linespacing=1.0,
        color=style.BLUE_RAMP[5],
    )
# the narrow run, one device on the core band and the top of the low-freq one
narrow = yaml.safe_load((BANDS / "narrow.yaml").read_text())["problem"]["f0_range"]
ax_share.annotate(
    "",
    (narrow[0] * 1e3, 0.70),
    (narrow[1] * 1e3, 0.70),
    xycoords=("data", "axes fraction"),
    arrowprops=dict(arrowstyle="|-|", color=style.ORANGE, lw=1.0, shrinkA=0, shrinkB=0, mutation_scale=2),
)
style.direct_label(ax_share, narrow[0] * 1e3, 0.70 * 1.6 * (visible / visible.sum() / dex).max(), "narrow run", dx=-3, color=style.ORANGE)

ax_share.set(
    xlim=tuple(10**log_range * 1e3),
    ylim=(0, 1.6 * (visible / visible.sum() / dex).max()),
    xlabel=r"$f_0$ [mHz]",
    ylabel="share per dex",
)
ax_share.set_title("Where the visible sources are")
style.panel_label(ax_share, "b")

style.save(fig, "narrow-band")
