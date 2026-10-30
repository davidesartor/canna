"""Sweep bench_micro.py's point over micro-batch sizes, then over accumulated batches, for each band x size.

Phase 1: batch m with no accumulation, m = 1, 2, 4, ... until it runs out of memory or passes
--max-micro. Phase 2: batches B = 2m*, 4m*, ... (m* the largest m that fit) accumulated over
B / m* micro-batches, halving the micro-batch on an OOM, while the predicted step stays under
--max-step seconds. One process for the whole sweep, so the imports and the GPU start are paid
once and each config is set up once; the allocator's peak only grows within a process, so memory
is XLA's plan. Runs already in the output are reused, so a requeued sweep resumes where it stopped.
While a point is timed, the next --compile-ahead points compile on CPU threads (default: half the
CPUs with autotuning off, none with it on, since autotuning runs kernels on the GPU).

Run from the repo root inside an allocation (see micro.sbatch):
    uv run --extra cuda --frozen python reports/benchmarks/step-time-memory/scripts/sweep_micro.py out.jsonl --bands narrow --sizes XS --max-micro 4 --max-batch 16
"""

import argparse
import gc
import json
import os
import traceback
from concurrent.futures import Future, ThreadPoolExecutor, wait
from pathlib import Path

import jax

import bench_micro


def recorded(out: Path, cfg: str, batch: int, micro: int) -> dict | None:
    for line in reversed(out.read_text().splitlines()):
        record = json.loads(line)
        if (
            record["config"] == cfg
            and record["batch"] == batch
            and record["micro_batches"] == micro
        ):
            return record
    return None


def sweep(
    out: Path,
    cfg: str,
    max_micro: int,
    max_batch: int,
    max_step: float,
    pool: ThreadPoolExecutor,
    ahead: int,
) -> None:
    prepared = None
    compiling: dict[tuple[int, int], Future] = {}

    def run(
        batch: int, micro: int, repeats: int, upcoming: list[tuple[int, int, int]]
    ) -> dict | None:
        """One point's record, or None when it failed for a reason other than OOM; `upcoming` compile meanwhile."""
        nonlocal prepared
        if (done := recorded(out, cfg, batch, micro)) is not None:
            return done
        try:
            if prepared is None:
                prepared = bench_micro.prepare(cfg, 3)
            for b, k, r in [(batch, micro, repeats), *upcoming[:ahead]]:
                if (b, k) not in compiling and recorded(out, cfg, b, k) is None:
                    compiling[b, k] = pool.submit(
                        bench_micro.compile_point, *prepared, b, k, r
                    )
            record = compiling.pop((batch, micro)).result()()
        except Exception:
            traceback.print_exc()
            print(f"{cfg} batch {batch} micro {micro}: failed", flush=True)
            return None
        finally:
            gc.collect()
        record["compile_ahead"] = ahead
        with open(out, "a") as fh:
            fh.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        return record

    try:
        best = None
        sizes = [2**i for i in range(min(max_micro, max_batch).bit_length())]
        for i, m in enumerate(sizes):
            record = run(m, 1, 3, [(s, 1, 3) for s in sizes[i + 1 :]])
            if record is None or "oom" in record:
                break
            best = record
        if best is None:
            return

        m = best["micro"]
        batches = []
        batch = 2 * m
        while batch <= max_batch:
            predicted = best["step_s"] * batch / m
            if predicted > max_step:
                print(
                    f"{cfg} batch {batch}: predicted {predicted:.0f} s per step, stopping",
                    flush=True,
                )
                break
            batches.append(batch)
            batch *= 2
        for i, batch in enumerate(batches):
            upcoming = [(b, b // m, 1) for b in batches[i + 1 :]]
            micro = m
            while micro >= 1:
                record = run(batch, batch // micro, 1, upcoming if micro == m else [])
                if record is not None and "oom" not in record:
                    break
                micro //= 2
            else:
                return
    finally:
        # drop this config's speculative compiles before the next one sets up
        for future in compiling.values():
            future.cancel()
        wait(compiling.values())
        compiling.clear()
        jax.clear_caches()
        gc.collect()


if __name__ == "__main__":
    autotune = "--xla_gpu_autotune_level=0" not in os.environ.get("XLA_FLAGS", "")
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("--bands", nargs="+", default=["narrow", "full"])
    parser.add_argument("--sizes", nargs="+", default=["XS", "S", "B", "L", "XL"])
    parser.add_argument("--max-batch", type=int, default=16384)
    parser.add_argument("--max-micro", type=int, default=16384)
    parser.add_argument("--max-step", type=float, default=20.0)
    parser.add_argument(
        "--compile-ahead",
        type=int,
        default=0 if autotune else int(os.environ.get("SLURM_CPUS_PER_TASK", 2)) // 2,
    )
    cli = parser.parse_args()
    cli.out.parent.mkdir(parents=True, exist_ok=True)
    cli.out.touch()
    print(jax.devices()[0].device_kind, flush=True)
    print(f"compile ahead: {cli.compile_ahead}", flush=True)
    with ThreadPoolExecutor(max(1, cli.compile_ahead)) as pool:
        for band in cli.bands:
            for size in cli.sizes:
                sweep(
                    cli.out,
                    f"{band}-{size}",
                    cli.max_micro,
                    cli.max_batch,
                    cli.max_step,
                    pool,
                    cli.compile_ahead,
                )
