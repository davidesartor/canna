"""Posterior-sampling throughput + peak-memory sweep across configs x ODE steps.

Each (config, ode_steps) runs in its own `_bench_worker` subprocess (PHASE=eval) so
peak GPU memory is the clean high-water of one posterior draw. Draws are fixed at
EVAL_DRAWS -- the sweep axis is the RK4 step count. Slow (one XLA warmup + compile
per combination) -- launch on a compute node
(`uv run pytest tests/test_eval_bench.py`).
"""

from canna import lisa

from _bench import CONFIGS, PEAK_BUDGET_MB, max_batch, run_worker
from _helpers import problem_name, save_text

EVAL_DRAWS = 64  # posterior samples per observation; fixed, not swept
ODE_STEP_SWEEP = [2, 4, 8, 16]


def test_eval_bench():
    header = (
        f"{'config':>8} {'ode steps':>9} {'compile ms':>11} "
        f"{'step ms':>9} {'peak (MB)':>12}"
    )
    lines = [
        f"{problem_name()}  n_sources={lisa.N_SOURCES}  (eval, draws={EVAL_DRAWS})",
        header,
    ]
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
    """Largest power-of-two draw count that fits in memory, per config."""
    fits = {cfg.name: max_batch(cfg.name, "eval") for cfg in CONFIGS}
    lines = [f"{problem_name()}  n_sources={lisa.N_SOURCES}  (eval max draws)"]
    lines += [f"{name:>8} {n:>8}" for name, n in fits.items()]
    save_text("eval_max_batch", "\n".join(lines))

    # a zero means not even one draw fit -- a real failure, not a benchmark result
    starved = [name for name, n in fits.items() if n == 0]
    assert not starved, f"no draw count fits for: {', '.join(starved)}"
