"""Figure: wall time of flow posterior sampling per window against model size, next to the MCMC reference."""

import json

import numpy as np

import style
from style import plt

COST = style.REPO / "reports/benchmarks/sampling-cost/data"
SIZES = ("XS", "S", "B", "L", "XL")
DRAWS = {256: style.BLUE, 1024: style.AQUA}
TIME_TICKS = {
    0.1: "0.1 s",
    1: "1 s",
    10: "10 s",
    60: "1 min",
    600: "10 min",
    3600: "1 h",
    36000: "10 h",
}

flow = [
    json.loads(line)
    for line in (COST / "a100.jsonl").read_text().splitlines()
    if line.strip()
]
mcmc = json.loads((COST / "mcmc.json").read_text())
assert {r["ode_steps"] for r in flow} == {32}
best = {(r["config"], r["draws"]): r["best_s"] for r in flow}
params = {
    s: next(r["params"] for r in flow if r["config"] == f"full-{s}") / 1e6
    for s in SIZES
}
x = np.array(list(params.values()))

style.setup()
fig, ax = plt.subplots(figsize=(style.COLUMN, 2.4), layout="constrained")

# flow, best of three windows: the full band as a line, the narrow band as rings around it
for draws, color in DRAWS.items():
    full = np.array([best[f"full-{s}", draws] for s in SIZES])
    narrow = np.array([best[f"narrow-{s}", draws] for s in SIZES])
    ax.plot(x, full, color=color, marker="o", ms=2.6, mec="white", mew=0.5, zorder=3)
    ax.plot(
        x, narrow, ls="none", marker="o", ms=5, mfc="none", mec=color, mew=0.7, zorder=3
    )
    style.direct_label(ax, x[-1], full[-1], f"flow, {draws} draws", dx=5, color=color)

# the reference: median and range over the joint chains, one per injection (window)
seconds = np.array([chain["seconds"] for chain in mcmc["joint"]])
mcmc_median = np.median(seconds)
ax.axhspan(seconds.min(), seconds.max(), color=style.FILL, zorder=0)
ax.axhline(mcmc_median, color=style.INK2, lw=0.9, zorder=1)
style.direct_label(
    ax, x[0], seconds.max(), "MCMC, one window, 2 CPU cores", dx=0, dy=2, va="bottom"
)

# the gap at the B size, 256 draws
flow_b = best["full-B", 256]
ax.annotate(
    "",
    (params["B"], mcmc_median),
    (params["B"], flow_b),
    arrowprops=dict(arrowstyle="<->", color=style.INK2, lw=0.7, shrinkA=3, shrinkB=1),
)
style.direct_label(
    ax,
    params["B"],
    np.sqrt(mcmc_median * flow_b),
    rf"$\times 10^{{{np.log10(mcmc_median / flow_b):.1f}}}$",
)
style.direct_label(
    ax, 0.99, 0.03, "lines: full band, rings: narrow band", dx=0, ha="right", va="bottom", xycoords="axes fraction"
)

ax.set_xscale("log")
ax.set_xticks(x, [f"{s}\n{p:.0f}M" for s, p in params.items()])
ax.set_xticks([], minor=True)
ax.set_xlabel("model size (parameters)")
ax.set_xlim(x[0] / 1.3, x[-1] * 3.4)
ax.set_yscale("log")
ax.set_yticks(list(TIME_TICKS), list(TIME_TICKS.values()))
ax.set_yticks([], minor=True)
ax.set_ylim(0.035, 1e5)
ax.set_ylabel("wall time per window")
style.save(fig, "sampling-cost")
