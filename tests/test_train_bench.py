"""Train-step throughput + peak-memory sweep across configs x batch sizes.

Each (config, batch) runs in its own `_bench_worker` subprocess (PHASE=train) so
peak GPU memory is the clean high-water of one train step. Slow (one XLA warmup +
compile per combination) -- launch on a compute node
(`uv run pytest tests/test_train_bench.py`).
"""

from canna import lisa

from _bench import (
    BATCH_SIZES,
    CONFIG_BY_NAME,
    CONFIGS,
    PEAK_BUDGET_MB,
    max_batch,
    run_worker,
)
from _helpers import problem_name, save_text

GEN_STEP_CONFIG = "B"  # train.py default width, for the gen-vs-step weighting


def test_train_bench():
    header = (
        f"{'config':>8} {'batch':>6} {'compile ms':>11} "
        f"{'step ms':>9} {'peak (MB)':>12}"
    )
    lines = [f"{problem_name()}  n_sources={lisa.N_SOURCES}  (train)", header]
    for cfg in CONFIGS:
        for batch in BATCH_SIZES:
            r = run_worker(cfg.name, batch, "train")
            assert (
                r["peak_mb"] < PEAK_BUDGET_MB
            ), f"{cfg.name}/batch={batch} train peak {r['peak_mb']:.0f}MB > {PEAK_BUDGET_MB}MB"
            lines.append(
                f"{r['config']:>8} {r['batch']:>6} {r['compile_ms']:>11.0f} "
                f"{r['step_ms']:>9.1f} {r['peak_mb']:>12.0f}"
            )
    save_text("train_bench", "\n".join(lines))


def test_batch_gen_bench():
    """Physics batch-gen cost vs train-step cost, and gen's share of the per-iter wall time."""
    cfg = CONFIG_BY_NAME[GEN_STEP_CONFIG]
    header = (
        f"{'batch':>6} {'gen ms':>9} {'step ms':>9} {'iter ms':>9} "
        f"{'gen %':>7} {'gen peak':>10} {'step peak':>10}"
    )
    lines = [
        f"{problem_name()}  n_sources={lisa.N_SOURCES}  "
        f"(gen vs step, config={GEN_STEP_CONFIG})",
        header,
    ]
    for batch in BATCH_SIZES:
        g = run_worker(cfg.name, batch, "gen")
        s = run_worker(cfg.name, batch, "train")
        iter_ms = g["step_ms"] + s["step_ms"]
        lines.append(
            f"{batch:>6} {g['step_ms']:>9.1f} {s['step_ms']:>9.1f} {iter_ms:>9.1f} "
            f"{100 * g['step_ms'] / iter_ms:>6.0f}% "
            f"{g['peak_mb']:>10.0f} {s['peak_mb']:>10.0f}"
        )
    save_text("batch_gen_bench", "\n".join(lines))


def test_train_max_batch():
    """Largest train batch that fits in memory, per config."""
    fits = {cfg.name: max_batch(cfg.name, "train") for cfg in CONFIGS}
    lines = [f"{problem_name()}  n_sources={lisa.N_SOURCES}  (train max batch)"]
    lines += [f"{name:>8} {n:>8}" for name, n in fits.items()]
    save_text("train_max_batch", "\n".join(lines))

    # a zero means not even one sample fit -- a real failure, not a benchmark result
    starved = [name for name, n in fits.items() if n == 0]
    assert not starved, f"no batch size fits for: {', '.join(starved)}"
