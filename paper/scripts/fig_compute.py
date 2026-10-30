"""Figure: step time, bf16 speed-up and memory plan of every LisaFlow size on an A100 and an L40S."""

import json

import numpy as np

import style
from style import plt

STEP = style.REPO / "reports/benchmarks/step-time-memory/data"
SIZES = ("XS", "S", "B", "L", "XL")
BANDS = ("narrow", "full")
CHIPS = {"a100": ("A100", style.BLUE), "l40s": ("L40S", style.ORANGE)}
SWEEP_BATCH = 128
LINE = {"full": "-", "narrow": (0, (3, 2))}


def records(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def size_axis(ax, params_by_size: dict) -> None:
    """Log parameter axis, ticked at each size with its count."""
    ax.set_xscale("log")
    ax.set_xticks(
        list(params_by_size.values()),
        [f"{s}\n{p:.0f}M" for s, p in params_by_size.items()],
    )
    ax.set_xticks([], minor=True)
    ax.set_xlabel("model size (parameters)")


def band_marker(band: str, color: str) -> dict:
    """Full band filled, narrow band hollow."""
    face = color if band == "full" else "white"
    return dict(marker="o", ms=3.2, mfc=face, mec=color, mew=0.9)


# the default-batch check (today's network): per config the unsplit plan, and the step of the fewest micro-batches that ran
check = {}
for chip in CHIPS:
    for r in records(STEP / f"{chip}-bf16backbone-b256.jsonl"):
        entry = check.setdefault((chip, r["config"]), {"params": r["params"]})
        if r["micro_batches"] == 1:
            entry["plan"] = r["plan_gib"]
            entry["limit"] = r["limit_gib"]
            entry["draw_plan"] = r.get("draw_plan_gib")
        if r.get("step_s") and r["micro_batches"] < entry.get("micro", 1 << 30):
            entry["step"], entry["micro"] = r["step_s"], r["micro_batches"]

# the precision sweep (an earlier revision of the network): fp32 over bf16 step at the sweep's batch
sweep = {}
for chip in CHIPS:
    for precision, suffix in (
        ("none", ""),
        ("attention", "-bf16attention"),
        ("backbone", "-bf16backbone"),
    ):
        for band in BANDS:
            for r in records(STEP / f"{chip}{suffix}-{band}.jsonl"):
                if (
                    r["batch"] == SWEEP_BATCH
                    and r["micro_batches"] == 1
                    and r.get("step_s")
                ):
                    sweep[chip, precision, r["config"]] = (r["params"], r["step_s"])

PARAMS_BY_SIZE = {s: check["a100", f"full-{s}"]["params"] / 1e6 for s in SIZES}

style.setup()
fig, axes = plt.subplots(1, 3, figsize=(style.FULL, 2.2), constrained_layout=True)

# (a) step time at the default batch
ax = axes[0]
for chip, (name, color) in CHIPS.items():
    for band in BANDS:
        rows = [
            check[chip, f"{band}-{s}"]
            for s in SIZES
            if "step" in check[chip, f"{band}-{s}"]
        ]
        params = np.array([r["params"] for r in rows]) / 1e6
        ax.plot(
            params,
            [r["step"] for r in rows],
            color=color,
            ls=LINE[band],
            **band_marker(band, color),
        )
        for p, r in zip(params, rows):
            if r["micro"] > 1:
                ax.annotate(
                    f"{r['micro']} micro-\nbatches",
                    (p, r["step"]),
                    xytext=(-6, 3),
                    textcoords="offset points",
                    ha="right",
                    va="bottom",
                    fontsize=6,
                    color=style.INK2,
                )
ax.set(yscale="log", ylabel="step [s]")
ax.set_yticks([0.03, 0.1, 0.3, 1, 3], ["0.03", "0.1", "0.3", "1", "3"])
ax.set_yticks([], minor=True)
size_axis(ax, PARAMS_BY_SIZE)
handles = [plt.Line2D([], [], color=c, lw=1.4) for _, c in CHIPS.values()]
handles += [
    plt.Line2D([], [], color=style.INK2, ls=LINE[b], **band_marker(b, style.INK2))
    for b in BANDS
]
ax.legend(
    handles,
    [n for n, _ in CHIPS.values()] + [f"{b} band" for b in BANDS],
    loc="upper left",
)
style.panel_label(ax, "a")

# (b) speed-up of bf16 over fp32, same network and batch
ax = axes[1]
for chip, (name, color) in CHIPS.items():
    for band in BANDS:
        for precision in ("backbone", "attention"):
            points = [
                (
                    sweep[chip, precision, c][0] / 1e6,
                    sweep[chip, "none", c][1] / sweep[chip, precision, c][1],
                )
                for c in (f"{band}-{s}" for s in SIZES)
                if (chip, precision, c) in sweep and (chip, "none", c) in sweep
            ]
            params, gain = np.array(points).T
            if precision == "backbone":
                ax.plot(
                    params, gain, color=color, ls=LINE[band], **band_marker(band, color)
                )
            else:
                ax.plot(
                    params,
                    gain,
                    color=style.VIOLET,
                    ls=LINE[band],
                    lw=0.8,
                    marker="x",
                    ms=2.5,
                    mew=0.8,
                )
ax.text(
    14,
    1.4,
    "bf16 attention only",
    ha="left",
    va="bottom",
    fontsize=6.5,
    color=style.VIOLET,
)
ax.text(
    14,
    45,
    "bf16 backbone",
    ha="left",
    va="center",
    fontsize=6.5,
    color=style.INK2,
)
ax.set(yscale="log", ylabel="speed-up over fp32", ylim=(0.8, 70))
size_axis(ax, PARAMS_BY_SIZE)
ax.set_yticks([1, 3, 10, 30], ["1", "3", "10", "30"])
ax.set_yticks([], minor=True)
style.panel_label(ax, "b")

# (c) planned device memory at the default batch, unsplit, against each card's cap
ax = axes[2]
for chip, (name, color) in CHIPS.items():
    for band in BANDS:
        rows = [check[chip, f"{band}-{s}"] for s in SIZES]
        params = np.array([r["params"] for r in rows]) / 1e6
        ax.plot(
            params,
            [r["plan"] for r in rows],
            color=color,
            ls=LINE[band],
            **band_marker(band, color),
        )
    limit = check[chip, "full-B"]["limit"]
    ax.axhline(limit, color=color, lw=0.7, ls=(0, (1, 1.5)))
    ax.text(14, limit * 1.06, f"{name} cap", fontsize=6, color=color, va="bottom")
draw_plan = check["a100", "full-B"]["draw_plan"]
ax.axhline(draw_plan, color=style.MUTED, lw=0.7, ls=(0, (1, 1.5)))
ax.text(
    14,
    draw_plan * 1.06,
    "full-band batch draw",
    fontsize=6,
    color=style.MUTED,
    va="bottom",
)
ax.set(yscale="log", ylabel="planned memory [GiB]")
size_axis(ax, PARAMS_BY_SIZE)
ax.set_yticks([1, 3, 10, 30], ["1", "3", "10", "30"])
ax.set_yticks([], minor=True)
style.panel_label(ax, "c")

style.save(fig, "compute")
