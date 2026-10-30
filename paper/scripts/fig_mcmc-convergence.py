"""Figure: convergence of the joint reference chains: burn-in drift, ESS and R-hat per long-chain injection."""

import json

import numpy as np
from matplotlib.ticker import FuncFormatter, NullFormatter

import style
from style import plt

DATA = style.REPO / "reports/mcmc/data/convergence"
ITERATIONS = FuncFormatter(lambda x, _: f"{x / 1000:g}k" if x else "0")

summary = json.loads((DATA / "summary.json").read_text())
curves = np.load(DATA / "curves.npz")
criterion = summary["criterion"]
windows = sorted(summary["windows"], key=lambda j: -summary["windows"][j]["window_snr"])
paper = summary["windows"][windows[0]]["settings"]["paper"]
colors = dict(zip(windows, [style.BLUE, style.ORANGE, style.GREEN, style.VIOLET]))

style.setup()
fig, (ax_drift, ax_ess, ax_rhat) = plt.subplots(
    1, 3, figsize=(style.FULL, 2.05), constrained_layout=True
)

# burn-in: the worst parameter's drift in each segment, in units of the tolerance
for j in windows:
    drift = np.maximum(
        curves[f"w{j}_burn_window_median"] / criterion["median_tol"],
        curves[f"w{j}_burn_window_width"] / criterion["width_tol"],
    ).max(axis=1)
    ax_drift.plot(curves[f"w{j}_burn_window"], drift, color=colors[j], marker="o")
ax_drift.axhline(1, **style.GUIDE)
ax_drift.axvline(paper["burn"], **style.GUIDE)
ax_drift.set_yscale("log")
ax_drift.set_xlabel("segment start [iterations]")
ax_drift.set_ylabel("worst drift / tolerance")
ax_drift.xaxis.set_major_formatter(ITERATIONS)
ax_drift.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))
ax_drift.yaxis.set_minor_formatter(NullFormatter())
style.direct_label(
    ax_drift,
    paper["burn"],
    0.97,
    "burn-in used",
    va="top",
    xycoords=("data", "axes fraction"),
)

# kept length: worst ESS and R-hat over parameters, after the burn-in used
for j in windows:
    kept = curves[f"w{j}_keep_keep"]
    ax_ess.plot(kept, curves[f"w{j}_keep_ess"].min(axis=1), color=colors[j], marker="o")
    ax_rhat.plot(
        kept,
        np.abs(curves[f"w{j}_keep_rhat"] - 1).max(axis=1),
        color=colors[j],
        marker="o",
    )
ax_ess.axhline(criterion["ess_min"], **style.GUIDE)
ax_rhat.axhline(criterion["rhat_max"] - 1, **style.GUIDE)
ax_ess.set_ylabel(r"min ESS per seed")
ax_rhat.set_ylabel(r"max $|\hat R - 1|$")
for ax in (ax_ess, ax_rhat):
    ax.axvline(paper["keep"], **style.GUIDE)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("kept iterations")
    ax.xaxis.set_major_formatter(ITERATIONS)
    ax.xaxis.set_minor_formatter(NullFormatter())
style.direct_label(
    ax_ess,
    ax_ess.get_xlim()[0],
    criterion["ess_min"],
    f"ESS {criterion['ess_min']:,}",
    dx=2,
    dy=5,
)
style.direct_label(
    ax_rhat,
    ax_rhat.get_xlim()[0],
    criterion["rhat_max"] - 1,
    rf"$\hat R - 1 = {criterion['rhat_max'] - 1:g}$",
    dx=2,
    dy=5,
)
for ax, letter in zip((ax_drift, ax_ess, ax_rhat), "abc"):
    style.panel_label(ax, letter)
fig.legend(
    [plt.Line2D([], [], color=colors[j], marker="o") for j in windows],
    [f"injection {j}, SNR {summary['windows'][j]['window_snr']:.0f}" for j in windows],
    loc="outside upper center",
    ncol=len(windows),
)
style.save(fig, "mcmc-convergence")
