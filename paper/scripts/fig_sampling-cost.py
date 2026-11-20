"""Figure: wall time of flow posterior sampling per window against model size, next to the MCMC reference."""

import json

import numpy as np

import style
from style import plt

COST = style.REPO / "reports/benchmarks/sampling-cost/data"
SIZES = ("XS", "S", "B", "L", "XL")
DRAWS = (256, 1024)
FLOW, MCMC = style.ORANGE, style.BLUE
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
fig, ax = plt.subplots(figsize=(style.COLUMN, 2.5), layout="constrained")

# flow, best of three windows: the full band filled, the narrow band hollow
for draws in DRAWS:
    for band, face in (("full", FLOW), ("narrow", "white")):
        t = np.array([best[f"{band}-{s}", draws] for s in SIZES])
        ax.plot(x, t, color=FLOW, marker="o", ms=3.2, mfc=face, mec=FLOW, mew=0.8, lw=1.0, zorder=3, label=f"flow, {band} band")
    above = draws == max(DRAWS)
    style.direct_label(ax, x[1], t[1], f"{draws} draws", dx=0, dy=7 if above else -7, ha="center", va="bottom" if above else "top")

# the reference: median and range over the joint chains, one per injection (window)
seconds = np.array([chain["seconds"] for chain in mcmc["joint"]])
ax.axhspan(seconds.min(), seconds.max(), color=MCMC, alpha=0.15, lw=0, zorder=0)
ax.axhline(np.median(seconds), color=MCMC, lw=1.0, zorder=1)
style.direct_label(
    ax, x[0] / 1.2, seconds.max(), "joint MCMC", dx=0, dy=3, va="bottom", color=MCMC
)

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[:2], labels[:2], loc="center left", bbox_to_anchor=(0.0, 0.5))

ax.set_xscale("log")
ax.set_xticks(x, [f"{s}\n{p:.0f}M" for s, p in params.items()])
ax.set_xticks([], minor=True)
ax.set_xlabel("model size (parameters)")
ax.set_xlim(x[0] / 1.3, x[-1] * 1.3)
ax.set_yscale("log")
ax.set_yticks(list(TIME_TICKS), list(TIME_TICKS.values()))
ax.set_yticks([], minor=True)
ax.set_ylim(0.035, 1e5)
ax.set_ylabel("wall time per window")
style.save(fig, "sampling-cost")
