"""The simulator against published references: TDI X against Babak et al., and the noise units against Robson et al."""

import json

import matplotlib.pyplot as plt
import numpy as np

import style

DATA = style.REPO / "reports/diagnostics/forward-model/data"
BINS = 28


def binned_quantiles(
    x: np.ndarray, y: np.ndarray, edges: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Bin centres (geometric) and the 5/50/95% quantiles of y per non-empty bin."""
    which = np.digitize(x, edges) - 1
    centres, rows = [], []
    for b in range(len(edges) - 1):
        in_bin = y[which == b]
        if in_bin.size >= 8:
            centres.append(np.sqrt(edges[b] * edges[b + 1]))
            rows.append(np.quantile(in_bin, [0.05, 0.5, 0.95]))
    return np.array(centres), np.array(rows)


style.setup()
record = json.loads((DATA / "forward-model.json").read_text())
tdi = np.load(DATA / "tdi.npz")
sky = np.load(DATA / "sky.npz")

fig, (ax_tdi, ax_sky) = plt.subplots(
    1, 2, figsize=(style.FULL, 2.3), layout="constrained"
)

# (a) TDI X: relative waveform difference and power deviation, binned over f0
f0_mhz = tdi["f0"] * 1e3
edges = np.geomspace(f0_mhz.min(), f0_mhz.max() * 1.0001, BINS + 1)
for values, color, name in (
    (tdi["rel"], style.BLUE, r"$\|X_{\rm B} - X\| / \|X\|$"),
    (np.abs(tdi["power_dev"]), style.ORANGE, r"$|\,\|X_{\rm B}\|^2 / \|X\|^2 - 1\,|$"),
):
    centres, q = binned_quantiles(f0_mhz, values, edges)
    ax_tdi.fill_between(centres, q[:, 0], q[:, 2], color=color, alpha=0.18, lw=0)
    ax_tdi.plot(centres, q[:, 1], color=color, label=name)
high_mhz = (record["tdi_high_lo_hz"] + record["tdi_high_hi_hz"]) / 2 * 1e3
cubic = np.geomspace(0.12, 12, 20)
ax_tdi.plot(
    cubic, record["tdi_rel_median_high"] * (cubic / high_mhz) ** 3, **style.GUIDE
)
style.direct_label(
    ax_tdi,
    0.5,
    record["tdi_rel_median_high"] * (0.5 / high_mhz) ** 3,
    r"$\propto f_0^{\,3}$",
    dx=-2,
    dy=7,
    ha="right",
)
low_mhz = record["tdi_low_hz"] * 1e3
ax_tdi.hlines(record["tdi_rel_max_low"], f0_mhz.min(), low_mhz, color=style.INK, lw=0.8)
style.direct_label(
    ax_tdi,
    np.sqrt(f0_mhz.min() * low_mhz),
    record["tdi_rel_max_low"],
    rf"max below {low_mhz:g} mHz",
    dx=0,
    dy=5,
    ha="center",
)
ax_tdi.set(
    xscale="log",
    yscale="log",
    xlabel=r"$f_0$ [mHz]",
    ylabel="difference to Babak et al.",
    ylim=(1e-10, 0.05),
)
ax_tdi.legend(loc="upper left")
ax_tdi.set_title("TDI-1.5 X against the closed form")
style.panel_label(ax_tdi, "a")

# (b) sky-averaged sensitivity over Robson et al.'s, in both units
f_mhz = sky["f"] * 1e3
ax_sky.axhline(0.5, **style.GUIDE)
ax_sky.plot(f_mhz, sky["ratio_length"], color=style.MUTED, marker="o", ms=2.4, mew=0)
ax_sky.plot(f_mhz, sky["ratio"], color=style.BLUE, marker="o", ms=2.4, mew=0)
style.direct_label(
    ax_sky,
    f_mhz[2],
    sky["ratio"][2],
    "fractional frequency (ours)",
    dx=0,
    dy=7,
    ha="left",
)
style.direct_label(
    ax_sky,
    f_mhz[4],
    sky["ratio_length"][4],
    rf"fractional length, slope ${record['sky_slope_length']:.1f}$".replace("-", "−"),
    dx=5,
    dy=3,
    va="bottom",
)
ax_sky.set(
    xscale="log",
    yscale="log",
    xlabel=r"$f$ [mHz]",
    ylabel=r"sky-averaged $S_n$ / Robson et al.",
    yticks=[0.1, 0.5, 1, 10, 100, 1000],
    yticklabels=["0.1", "1/2", "1", "10", "100", "1000"],
)
ax_sky.set_title("Sky-averaged sensitivity")
style.panel_label(ax_sky, "b")

style.save(fig, "forward-model-checks")
