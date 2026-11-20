"""Figures: fit quality against training time (cooldown forks of each run), and MCMC windows to break even with the flow."""

import json
import math

import numpy as np
import yaml

import style
from style import plt

DATA = style.PAPER / "data/runs/source-count.json"
COST = style.REPO / "reports/benchmarks/sampling-cost/data/a100.jsonl"
BANDS = style.REPO / "src/canna/lisa/configs/band"
INTERIOR_RESPONSES = 3  # a window's interior holds 3R bins (ForwardModelInteriorResponses)
CONFIG_STYLE = {
    ("narrow", 2): (style.BLUE, "-", "narrow, 2"),
    ("narrow", 4): (style.AQUA, "-", "narrow, 4"),
    ("narrow", 8): (style.VIOLET, "-", "narrow, 8"),
    ("full", 4): (style.ORANGE, (0, (4, 2)), "full, 4"),
    ("full", 8): (style.RED, (0, (4, 2)), "full, 8"),
    ("full", 16): (style.YELLOW, (0, (4, 2)), "full, 16"),
}
BAND_STYLE = {"narrow": style.BLUE, "full": style.ORANGE}
FORKS = (4, 8, 16, 32)
VISIBLE = ("7-15", "15-40", "40-100", ">=100", "100-inf")
MCMC_SECONDS = np.logspace(0, np.log10(7 * 86400), 200)
HOUR_TICKS = (2.5, 5, 10, 20, 40, 80)
WIDTH_TICKS = (1, 2, 4, 8)
TIME_TICKS = {1: "1 s", 60: "1 min", 3600: "1 h", 86400: "1 d", 7 * 86400: "1 wk"}

runs = {r["run"]: r for r in json.loads(DATA.read_text())["runs"]}


def hours(run: dict) -> float:
    """A100 hours of steps of a run trained from scratch to its planned length."""
    return (
        run["epochs_planned"]
        * run["log_interval"]
        / 1000
        * run["smoke_epoch_s_per_1000"]
        / 3600
    )


def family(band: str, n: int) -> list[dict]:
    """The cooldown forks and the 64-epoch run (full S=4: its Student-t continuation), shortest first."""
    names = [f"261103-n{n}-{band}-B-e{k}" for k in FORKS]
    names.append(
        "261103-n4-full-B-t44" if (band, n) == ("full", 4) else f"261103-n{n}-{band}-B"
    )
    return [runs[m] for m in names if m in runs and runs[m]["evals"]]


def width_ratio(ev: dict) -> float:
    return next(x for x in ev["vs_mcmc"] if x["band"] == "15-40")["width_ratio"][
        "f0 [bins]"
    ]


def coverage68(ev: dict) -> float:
    """Share of true flow coordinates in the central 68% interval, pooled over the visible sources and coordinates."""
    rows = [x for x in ev["scores"] if x["band"] in VISIBLE]
    return sum(np.mean(x["in68"]) * x["n"] for x in rows) / sum(x["n"] for x in rows)


style.setup()

# ---- quality against training time ---------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(style.FULL, 2.3), layout="constrained")
for (band, n), (color, ls, name) in CONFIG_STYLE.items():
    fam = family(band, n)
    h = np.array([hours(r) for r in fam])
    evs = [r["evals"][-1] for r in fam]
    # the Student-t continuation's logged loss does not compare with the Gaussian forks'
    nll = [
        np.nan if r["run"].endswith("-t44") else ev["catalog_at_eval"]
        for r, ev in zip(fam, evs)
    ]
    for ax, y in zip(
        axes, (nll, [width_ratio(ev) for ev in evs], [coverage68(ev) for ev in evs])
    ):
        ax.plot(
            h,
            y,
            color=color,
            ls=ls,
            marker="o",
            ms=2.8,
            mec="white",
            mew=0.5,
            lw=1.0,
            label=name,
        )
axes[0].set(ylabel="eval-catalogue NLL")
axes[0].set_title("Loss", loc="left")
axes[1].set(ylabel="flow / MCMC width", yscale="log")
axes[1].set_title(r"$f_0$ width, SNR 15–40", loc="left")
axes[1].set_yticks(WIDTH_TICKS, [f"{t:g}" for t in WIDTH_TICKS])
axes[1].set_yticks([], minor=True)
axes[1].axhline(1.0, **style.GUIDE)
axes[2].set(ylabel="coverage")
axes[2].set_title(r"68% coverage, SNR $\geq$ 7", loc="left")
axes[2].axhline(0.68, **style.GUIDE)
axes[0].legend(frameon=False, fontsize=6, ncol=2)
for ax, letter in zip(axes, "abc"):
    ax.set_xscale("log")
    ax.set_xticks(HOUR_TICKS, [f"{t:g}" for t in HOUR_TICKS])
    ax.set_xticks([], minor=True)
    ax.set_xlabel("training time [A100 h]")
    style.panel_label(ax, letter)
style.save(fig, "amortisation-quality")

# ---- break-even windows against MCMC time per window -------------------------------------------------------
flow_s = {
    r["config"].split("-")[0]: r["best_s"]
    for r in map(json.loads, COST.read_text().splitlines())
    if r["config"] in ("narrow-B", "full-B") and r["draws"] == 256
}


def break_even(train_hours: float, mcmc_s: np.ndarray, band: str) -> np.ndarray:
    """Windows after which training plus flow sampling costs no more than MCMC on every window."""
    gap = mcmc_s - flow_s[band]
    return np.where(gap > 0, 3600 * train_hours / np.where(gap > 0, gap, 1), np.nan)


def windows_per_pass(band: str) -> int:
    """Windows that tile the band once, each contributing its interior."""
    problem = yaml.safe_load((BANDS / f"{band}.yaml").read_text())["problem"]
    bins = (problem["f0_range"][1] - problem["f0_range"][0]) * problem["t_obs"]
    return math.ceil(bins / (INTERIOR_RESPONSES * problem["response_points"]))


fig, ax = plt.subplots(figsize=(style.COLUMN, 2.6), layout="constrained")
line_labels = []
for band, color in BAND_STYLE.items():
    budgets = [hours(r) for b, n in CONFIG_STYLE if b == band for r in family(b, n)]
    shortest, longest = min(budgets), max(budgets)
    ax.fill_between(MCMC_SECONDS, break_even(shortest, MCMC_SECONDS, band), break_even(longest, MCMC_SECONDS, band),
                    color=color, alpha=0.15, lw=0)
    ax.plot(MCMC_SECONDS, break_even(longest, MCMC_SECONDS, band), color=color, lw=1.0)
    ax.plot(MCMC_SECONDS, break_even(shortest, MCMC_SECONDS, band), color=color, lw=0.7, ls=(0, (2, 1.5)))
    line_labels.append((band, color, longest))
    # one pass over the band, and the MCMC time per window at which that pass pays for the training
    passes = windows_per_pass(band)
    ax.axhline(passes, **style.GUIDE)
    style.direct_label(ax, MCMC_SECONDS[-1] / 1.3, passes, f"one pass, {band} ({passes:,})", dx=0, dy=2,
                       ha="right", va="bottom")
    pays = 3600 * longest / passes + flow_s[band]
    ax.plot(pays, passes, marker="o", ms=3.2, mec="white", mew=0.5, color=color, zorder=3)

ax.set(xscale="log", yscale="log", xlim=MCMC_SECONDS[[0, -1]], ylim=(1, None), xlabel="MCMC wall time per window",
       ylabel="windows to break even")
ax.set_xticks(list(TIME_TICKS), list(TIME_TICKS.values()))
ax.set_xticks([], minor=True)

# label each band's solid line along it, the full band above and the narrow band below
fig.canvas.draw()
LABEL_AT = 600
for band, color, longest in line_labels:
    x = np.array([LABEL_AT, 2 * LABEL_AT])
    (x0, y0), (x1, y1) = ax.transData.transform(np.column_stack([x, break_even(longest, x, band)]))
    angle = np.arctan2(y1 - y0, x1 - x0)
    side = 1 if band == "full" else -1
    style.direct_label(ax, LABEL_AT, break_even(longest, x[:1], band)[0], f"{band}, {longest:.0f} h",
                       dx=-2 * side * np.sin(angle), dy=2 * side * np.cos(angle), ha="center",
                       va="bottom" if side > 0 else "top", color=color, rotation=np.degrees(angle),
                       rotation_mode="anchor")
style.save(fig, "amortisation-break-even")
