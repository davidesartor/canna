"""Per-stage wall-clock of one training epoch, per run config, per GPU (GPU node)."""

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
    RUN_CONFIGS,
    STAGES,
    append_csv,
    device_kind,
    run_config_args,
    run_stage_worker,
    save_text,
)


@pytest.mark.parametrize("run_config", RUN_CONFIGS)
def test_stage_bench(run_config):
    args = run_config_args(run_config)
    rows = [run_stage_worker(run_config, args.batch_size, stage) for stage in STAGES]
    append_csv(rows)

    # an epoch is log_interval x (gen + train) plus one ckpt and plot; train_fused
    # is reported per-step already, purely for comparison, and excluded from the total
    by_stage = {r["phase"]: r["step_ms"] for r in rows}
    epoch_ms = args.log_interval * (by_stage["gen"] + by_stage["train"])
    epoch_ms += by_stage["ckpt"] + by_stage["plot"]

    header = f"{'stage':>8} {'compile ms':>11} {'step ms':>9} {'per epoch %':>12} {'peak (MB)':>10}"
    lines = [f"{run_config} on {device_kind()} (batch={args.batch_size})", header]
    for r in rows:
        if r["phase"] == "train_fused":
            lines.append(
                f"{r['phase']:>8} {r['compile_ms']:>11.0f} {r['step_ms']:>9.1f} "
                f"{'--':>11} {r['peak_mb']:>10.0f}"
            )
            continue
        reps = args.log_interval if r["phase"] in ("gen", "train") else 1
        lines.append(
            f"{r['phase']:>8} {r['compile_ms']:>11.0f} {r['step_ms']:>9.1f} "
            f"{100 * reps * r['step_ms'] / epoch_ms:>11.0f}% {r['peak_mb']:>10.0f}"
        )
    lines.append(f"epoch total {epoch_ms / 1e3:.1f}s ({args.log_interval} steps)")
    save_text(f"{device_kind()}_stage_bench_{run_config}", "\n".join(lines))
