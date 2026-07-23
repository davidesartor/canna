"""Posterior-sampling throughput + peak-memory sweep across configs x ODE steps (GPU node)."""

import jax
import pytest

pytestmark = [
    pytest.mark.bench,
    pytest.mark.skipif(
        jax.default_backend() == "cpu",
        reason="throughput/peak-memory benchmarks need a GPU",
    ),
]

from _bench import CONFIGS, PEAK_BUDGET_MB, max_batch, run_worker, save_text

EVAL_DRAWS = 64
ODE_STEP_SWEEP = [2, 4, 8, 16]


def test_eval_bench():
    header = f"{'config':>8} {'ode steps':>9} {'compile ms':>11} {'step ms':>9} {'peak (MB)':>12}"
    lines = [f"eval sweep (draws={EVAL_DRAWS})", header]
    for cfg in CONFIGS:
        for steps in ODE_STEP_SWEEP:
            r = run_worker(cfg.name, EVAL_DRAWS, "eval", ode_steps=steps)
            assert (
                r["peak_mb"] < PEAK_BUDGET_MB
            ), f"{cfg.name}/ode_steps={steps} eval peak {r['peak_mb']:.0f}MB > {PEAK_BUDGET_MB}MB"
            lines.append(
                f"{r['config']:>8} {steps:>9} {r['compile_ms']:>11.0f} "
                f"{r['step_ms']:>9.1f} {r['peak_mb']:>12.0f}"
            )
    save_text("eval_bench", "\n".join(lines))


def test_eval_max_batch():
    fits = {cfg.name: max_batch(cfg.name, "eval") for cfg in CONFIGS}
    lines = ["eval max draws"] + [f"{name:>8} {n:>8}" for name, n in fits.items()]
    save_text("eval_max_batch", "\n".join(lines))
    starved = [name for name, n in fits.items() if n == 0]
    assert not starved, f"no draw count fits for: {', '.join(starved)}"
