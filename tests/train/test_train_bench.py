"""Train-step throughput + peak-memory sweep across configs x batch sizes (GPU node)."""

import jax
import pytest

pytestmark = [
    pytest.mark.bench,
    pytest.mark.skipif(
        jax.default_backend() == "cpu",
        reason="throughput/peak-memory benchmarks need a GPU",
    ),
]

from _bench import (
    BATCH_SIZES,
    CONFIG_BY_NAME,
    CONFIGS,
    PEAK_BUDGET_MB,
    max_batch,
    run_worker,
    save_text,
)

GEN_STEP_CONFIG = "B"


def test_train_bench():
    header = f"{'config':>8} {'batch':>6} {'compile ms':>11} {'step ms':>9} {'peak (MB)':>12}"
    lines = ["train step sweep", header]
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
    cfg = CONFIG_BY_NAME[GEN_STEP_CONFIG]
    header = f"{'batch':>6} {'gen ms':>9} {'step ms':>9} {'iter ms':>9} {'gen %':>7}"
    lines = [f"gen vs step (config={GEN_STEP_CONFIG})", header]
    for batch in BATCH_SIZES:
        g = run_worker(cfg.name, batch, "gen")
        s = run_worker(cfg.name, batch, "train")
        iter_ms = g["step_ms"] + s["step_ms"]
        lines.append(
            f"{batch:>6} {g['step_ms']:>9.1f} {s['step_ms']:>9.1f} {iter_ms:>9.1f} "
            f"{100 * g['step_ms'] / iter_ms:>6.0f}%"
        )
    save_text("batch_gen_bench", "\n".join(lines))


def test_train_max_batch():
    fits = {cfg.name: max_batch(cfg.name, "train") for cfg in CONFIGS}
    lines = ["train max batch"] + [f"{name:>8} {n:>8}" for name, n in fits.items()]
    save_text("train_max_batch", "\n".join(lines))
    starved = [name for name, n in fits.items() if n == 0]
    assert not starved, f"no batch size fits for: {', '.join(starved)}"
