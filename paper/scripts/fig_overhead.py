"""Figure: where an A100 step goes, the simulator's batch draw per device, and the walltime lost against the log interval."""

import importlib.util
import json

import numpy as np

import style
from style import plt

BENCH = style.REPO / "reports/benchmarks"
SIZES = ("XS", "S", "B", "L", "XL")
BANDS = ("narrow", "full")
LINE = {"full": "-", "narrow": (0, (3, 2))}
TRAIN_OBSERVATIONS = 256
SOURCES = 4
DEVICES = {
    "a100": ("A100 40 GB", style.BLUE),
    "1080ti": ("GTX 1080 Ti", style.AQUA),
    "cpu": ("CPU, 8 cores", style.YELLOW),
}
CHIPS = {"A100": style.BLUE, "L40S": style.ORANGE}
# full-B's 80 GB record shared its GPU with another process: its A100 curve is the 40 GB card alone
A100_CARD = {"narrow": "A100 80 GB", "full": "A100 40 GB"}
SPLIT = (
    ("network", style.BLUE_RAMP[1]),
    ("optimiser update", style.BLUE_RAMP[3]),
    ("batch draw", style.BLUE_RAMP[5]),
)


def records(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def band_marker(band: str, color: str) -> dict:
    """Full band filled, narrow band hollow."""
    face = color if band == "full" else "white"
    return dict(marker="o", ms=3.2, mfc=face, mec=color, mew=0.9)


# the A100 check at the default batch, unsplit
check = {
    r["config"]: r
    for r in records(BENCH / "step-time-memory/data/a100-bf16backbone-b256.jsonl")
    if r.get("step_s") and r["micro_batches"] == 1
}
configs = [f"{b}-{s}" for b in BANDS for s in SIZES if f"{b}-{s}" in check]

# the simulator's training batch (16 draws per observation) at the default sources
simulator = {
    device: [
        r
        for r in json.loads(
            (BENCH / f"batch-draw/data/{device}.json").read_text()
        )["records"]
        if r["kind"] == "train_sample" and r["sources"] == SOURCES
    ]
    for device in DEVICES
}

# the walltime model of training-overhead, over every record there; a later job's record of a config supersedes
spec = importlib.util.spec_from_file_location(
    "chunk_model", BENCH / "training-overhead/scripts/overhead.py"
)
chunk_model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(chunk_model)
latest = {}
for path in sorted((BENCH / "training-overhead/data").glob("*.jsonl")):
    for r in records(path):
        key = (r["config"], chunk_model.series_of(r))
        if key not in latest or int(r["job"]) > int(latest[key]["job"]):
            latest[key] = r

style.setup()
fig, axes = plt.subplots(
    1,
    3,
    figsize=(style.FULL, 2.3),
    constrained_layout=True,
    gridspec_kw=dict(width_ratios=[1.1, 1, 1]),
)

# (a) shares of the A100 step at the default batch
ax = axes[0]
y = (
    np.arange(len(configs))[::-1]
    + np.array([b == "narrow" for c in configs for b in [c.split("-")[0]]]) * 0.6
)
for yi, name in zip(y, configs):
    r = check[name]
    parts = dict(
        zip(
            [label for label, _ in SPLIT],
            [
                r["step_s"] - r["draw_s"] - r["optimizer_s"],
                r["optimizer_s"],
                r["draw_s"],
            ],
        )
    )
    left = 0.0
    for label, color in SPLIT:
        share = 100 * parts[label] / r["step_s"]
        ax.barh(
            yi,
            share,
            left=left,
            height=0.75,
            color=color,
            lw=0,
            label=label if name == configs[0] else None,
        )
        left += share
    ax.text(
        101.5,
        yi,
        f"{1e3 * r['step_s']:.0f} ms",
        va="center",
        fontsize=6,
        color=style.INK2,
    )
ax.set_yticks(y, [c.replace("-", " ") for c in configs])
ax.tick_params(axis="y", length=0)
ax.set(xlim=(0, 100), xlabel="share of the step [%]")
ax.grid(axis="y", visible=False)
ax.spines["left"].set_visible(False)
ax.legend(
    loc="lower center",
    bbox_to_anchor=(0.5, 1.0),
    ncol=3,
    handlelength=0.9,
    columnspacing=0.8,
    fontsize=6,
)
style.panel_label(ax, "a", x=-0.28)

# (b) batch draw time against the observations per batch, with the A100's B steps for scale
ax = axes[1]
for device, (name, color) in DEVICES.items():
    for band in BANDS:
        points = sorted(
            (r["n"], r["ms"]) for r in simulator[device] if r["band"] == f"{band}-B"
        )
        n, ms = np.array(points).T
        ax.plot(
            n,
            ms,
            color=color,
            ls=LINE[band],
            **band_marker(band, color),
            label=name if band == "full" else None,
        )
for band in BANDS:
    step_ms = 1e3 * check[f"{band}-B"]["step_s"]
    ax.axhline(step_ms, color=style.MUTED, lw=0.7, ls=LINE[band])
    ax.annotate(
        f"A100 80 GB {band}-B step",
        (13 if band == "narrow" else 20000, step_ms),
        xytext=(0, 2),
        textcoords="offset points",
        fontsize=6,
        color=style.MUTED,
        va="bottom",
        ha="left" if band == "narrow" else "right",
        bbox=dict(fc=ax.get_facecolor(), ec="none", pad=0.5),
        zorder=3,
    )
ax.axvline(TRAIN_OBSERVATIONS, color=style.AXIS, lw=0.7)
ax.set(
    xscale="log",
    yscale="log",
    xlabel="observations per batch (16 draws each)",
    ylabel="batch draw time",
)
ax.set_xticks([16, 64, 256, 1024, 4096, 16384], ["16", "64", "256", "1k", "4k", "16k"])
ax.set_xticks([], minor=True)
ax.set_yticks([3, 10, 100, 1e3, 1e4], ["3 ms", "10 ms", "0.1 s", "1 s", "10 s"])
ax.set_yticks([], minor=True)
ax.legend(loc="upper left")
style.panel_label(ax, "b")

# (c) share of a 24 h job lost outside saved steps, against the log interval, for size B with the bf16 backbone
ax = axes[2]
default_interval = None
for (config, series), r in sorted(latest.items()):
    card, precision = series.split(" · bf16 ")
    band = config.split("-")[0]
    chip = "A100" if card == A100_CARD.get(band) else card
    if (
        r.get("oom")
        or precision != "backbone"
        or not config.endswith("-B")
        or chip not in CHIPS
    ):
        continue
    costs = chunk_model.costs(r)
    intervals = [L for L in chunk_model.CURVE if L <= costs["total"]]
    lost = [100 * chunk_model.chunk(costs, L)["lost"] for L in intervals]
    ax.plot(intervals, lost, color=CHIPS[chip], ls=LINE[band], lw=1.2)
    best = min(
        (L for L in chunk_model.NICE if L <= costs["total"]),
        key=lambda L: chunk_model.chunk(costs, L)["lost"],
    )
    ax.plot(
        best,
        100 * chunk_model.chunk(costs, best)["lost"],
        **band_marker(band, CHIPS[chip]),
    )
    default_interval = r["log_interval"]
ax.axvline(default_interval, color=style.AXIS, lw=0.7)
ax.text(default_interval * 1.15, 0.15, "default", fontsize=6, color=style.MUTED)
ax.set(
    xscale="log",
    xlabel="epoch length [steps]",
    ylabel="walltime lost per 24 h job [%]",
    ylim=(0, 5),
)
handles = [plt.Line2D([], [], color=c, lw=1.4) for c in CHIPS.values()]
handles += [plt.Line2D([], [], color=style.INK2, ls=LINE[b], lw=1.2) for b in BANDS]
ax.legend(
    handles,
    list(CHIPS) + [f"{b}-B" for b in BANDS],
    loc="lower center",
    bbox_to_anchor=(0.5, 1.0),
    ncol=2,
    handlelength=1.4,
    columnspacing=0.8,
    fontsize=6,
)
style.panel_label(ax, "c")

style.save(fig, "overhead")
