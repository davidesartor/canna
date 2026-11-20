"""Figure: step time and memory plan of every LisaFlow size on an A100, both bands."""

import json

import style
from style import plt

STEP = style.REPO / "reports/benchmarks/step-time-memory/data"
SIZES = ("XS", "S", "B", "L", "XL")
BANDS = {"full": style.BLUE, "narrow": style.ORANGE}
LIMIT = dict(color=style.MUTED, lw=0.7, ls=(0, (1, 1.5)))


def size_axis(ax, params_by_size: dict) -> None:
    """Log parameter axis, ticked at each size with its count."""
    ax.set_xscale("log")
    ax.set_xticks(
        list(params_by_size.values()),
        [f"{s}\n{p:.0f}M" for s, p in params_by_size.items()],
    )
    ax.set_xticks([], minor=True)
    ax.set_xlabel("model size (parameters)")


# per config at the default batch: the unsplit plan, and the step of the fewest micro-batches that ran
check = {}
for line in (STEP / "a100-bf16backbone-b256.jsonl").read_text().splitlines():
    if not line.strip():
        continue
    r = json.loads(line)
    entry = check.setdefault(r["config"], {"params": r["params"] / 1e6})
    if r["micro_batches"] == 1:
        entry["plan"] = r["plan_gib"]
        entry["limit"] = r["limit_gib"]
        entry["draw_plan"] = r.get("draw_plan_gib")
    if r.get("step_s") and r["micro_batches"] < entry.get("micro", 1 << 30):
        entry["step"], entry["micro"] = r["step_s"], r["micro_batches"]

PARAMS_BY_SIZE = {s: check[f"full-{s}"]["params"] for s in SIZES}

style.setup()
fig, (ax_step, ax_memory) = plt.subplots(
    1, 2, figsize=(style.FULL, 2.3), constrained_layout=True
)

# (a) step time at the default batch
for band, color in BANDS.items():
    rows = [check[f"{band}-{s}"] for s in SIZES if "step" in check[f"{band}-{s}"]]
    ax_step.plot(
        [r["params"] for r in rows],
        [r["step"] for r in rows],
        color=color,
        marker="o",
        label=f"{band} band",
    )
    for r in rows:
        if r["micro"] > 1:
            style.direct_label(
                ax_step,
                r["params"],
                r["step"],
                f"{r['micro']} micro-batches",
                dx=-5,
                dy=2,
            )
ax_step.set(yscale="log", ylabel="training step [s]", ylim=(0.028, 3))
ax_step.set_yticks([0.03, 0.1, 0.3, 1], ["0.03", "0.1", "0.3", "1"])
ax_step.set_yticks([], minor=True)
size_axis(ax_step, PARAMS_BY_SIZE)
ax_step.legend(loc="upper left")
ax_step.set_title("Training step")
style.panel_label(ax_step, "a")

# (b) planned device memory of the unsplit step against the card's usable memory
for band, color in BANDS.items():
    rows = [check[f"{band}-{s}"] for s in SIZES]
    ax_memory.plot(
        [r["params"] for r in rows], [r["plan"] for r in rows], color=color, marker="o"
    )
left = min(PARAMS_BY_SIZE.values())
for level, name in (
    (check["full-B"]["limit"], "usable memory of the A100"),
    (check["full-B"]["draw_plan"], "simulated full-band batch alone"),
):
    ax_memory.axhline(level, **LIMIT)
    style.direct_label(ax_memory, left, level, name, dx=0, dy=2, va="bottom")
ax_memory.set(yscale="log", ylabel="planned memory [GiB]", ylim=(1, 110))
size_axis(ax_memory, PARAMS_BY_SIZE)
ax_memory.set_yticks([1, 3, 10, 30], ["1", "3", "10", "30"])
ax_memory.set_yticks([], minor=True)
ax_memory.set_title("Memory of one unsplit step")
style.panel_label(ax_memory, "b")

style.save(fig, "compute")
