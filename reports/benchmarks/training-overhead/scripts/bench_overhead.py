"""Time what a LISA run spends outside its steps: the trainer's own epoch loop with its async checkpoint, and the restart.

Each config is set up and compiled once and reused for its timings; overhead.sbatch gives each
config its own process, as a restart has. One JSON line per config, skipped when this
Slurm job already wrote it (a requeue). A stage that runs out of memory is recorded as such and
the config's later stages are skipped. Checkpoints are written under each --targets directory
and removed after.

Run from the repo root (see overhead.sbatch):
    BENCH_T0=$(date +%s.%N) .venv/bin/python reports/benchmarks/training-overhead/scripts/bench_overhead.py out.jsonl --configs narrow-XS full-B --targets outputs /tmp
"""

import argparse
import gc
import json
import math
import os
import shutil
import subprocess
import time
from pathlib import Path

T0 = float(os.environ.get("BENCH_T0", time.time()))

import warnings

warnings.filterwarnings("ignore")
import equinox as eqx
import jax
import jax.random as jr
import numpy as np

from canna.lisa import config
from canna.lisa.network import (
    LOSS_GROUPS,
    LOSS_LOG10_SNR,
    LOSS_SNR_BINS,
    LOSS_TIME_BINS,
    LOSS_TIME_DECADES,
)
from canna.lisa.train import (
    checkpoint_manager,
    diagnostics,
    eval_catalog,
    restore_checkpoint,
    save_checkpoint,
    setup,
    train_epoch,
)

GIB = 2**30


class OutOfMemory(Exception):
    pass


def is_oom(e: Exception) -> bool:
    return "RESOURCE_EXHAUSTED" in str(e) or "out of memory" in str(e).lower()


def wall(fn, *args):
    """fn(*args) blocked until ready, and its wall time."""
    start = time.perf_counter()
    out = jax.block_until_ready(fn(*args))
    return out, time.perf_counter() - start


def plan_gib(compiled) -> float:
    """XLA's planned device memory for a compiled call (arguments + outputs + temporaries)."""
    m = compiled.memory_analysis()
    return (
        m.argument_size_in_bytes
        + m.output_size_in_bytes
        + m.temp_size_in_bytes
        - m.alias_size_in_bytes
    ) / GIB


def write_history(
    run_dir: Path, histories: dict[str, np.ndarray], interval: int
) -> None:
    """The trainer's history.npz, as train.main writes it after every epoch."""
    done = len(histories["loss_history"])
    np.savez(
        run_dir / "history.tmp.npz",
        epoch=done,
        steps=np.arange(1, done + 1) * interval,
        losses=histories["loss_history"].astype(np.float32),
        loss_names=np.array(["frame", "flow", "grad_norm"]),
        catalog=histories["catalog_history"],
        catalog_names=np.array(
            ["frame_x", "flow_x", "frame_y", "flow_y", "w_rms", "z_x"]
        ),
        frame_zoom=histories["frame_history"][:, 0],
        frame_calibration=histories["frame_history"][:, 1],
        f0_fisher=histories["fisher_history"],
        flow_zoom=histories["flow_history"][:, 0],
        flow_calibration=histories["flow_history"][:, 1],
        groups=np.array(["f0", "fdot", "sky", "z+", "z-"]),
        snr_edges=np.linspace(*LOSS_LOG10_SNR, LOSS_SNR_BINS + 1),
        time_edges=LOSS_TIME_DECADES
        * (np.arange(LOSS_TIME_BINS + 1) / LOSS_TIME_BINS - 1),
    )
    os.replace(run_dir / "history.tmp.npz", run_dir / "history.npz")


def full_histories(total_steps: int, interval: int) -> dict[str, np.ndarray]:
    """The trainer's histories at their full size for this log interval."""
    epochs = total_steps // interval
    return dict(
        loss_history=np.full((epochs, interval, 3), 0.5),
        catalog_history=np.full((epochs, 6), 0.5),
        frame_history=np.full((epochs, 2, LOSS_SNR_BINS, len(LOSS_GROUPS)), 0.5),
        fisher_history=np.full((epochs, LOSS_SNR_BINS), 0.5),
        flow_history=np.full((epochs, 2, LOSS_TIME_BINS, len(LOSS_GROUPS)), 0.5),
    )


def evict(path: Path) -> None:
    """Drop path's files from this node's page cache, so the next read comes off the filesystem as on a fresh node."""
    for f in path.rglob("*"):
        if f.is_file():
            fd = os.open(f, os.O_RDONLY)
            os.fsync(fd)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            os.close(fd)


def git_s() -> float:
    """The trainer's source checks at start: HEAD and a clean src/."""
    start = time.perf_counter()
    for args in (["rev-parse", "HEAD"], ["status", "--porcelain", "--", "."]):
        subprocess.run(["git", "-C", "src", *args], capture_output=True, check=True)
    return time.perf_counter() - start


def bench(
    name: str,
    record: dict,
    targets: list[str],
    steps: int,
    repeats: int,
    loop_epochs: int,
    history_intervals: list[int],
) -> dict:
    """Every stage of one config; the record, with "oom" at the stage that ran out of memory."""
    cfg = config.load(name)
    record |= dict(
        config=cfg.name,
        bf16=cfg.bf16,
        batch=cfg.batch_size,
        micro_batches=cfg.micro_batches,
        draws=cfg.draws,
        total_steps=cfg.total_steps,
        log_interval=cfg.log_interval,
    )

    def stage(stage_name, fn):
        try:
            return fn()
        except Exception as e:
            if not is_oom(e):
                raise
            record["oom"] = f"{stage_name}: {str(e).splitlines()[0][:300]}"
            raise OutOfMemory from e

    (problem, tx, state), record["setup_s"] = stage("setup", lambda: wall(setup, cfg))
    record["params"] = sum(
        x.size for x in jax.tree.leaves(eqx.filter(state.flow, eqx.is_inexact_array))
    )
    record["state_gib"] = (
        sum(x.nbytes for x in jax.tree.leaves(eqx.filter(state, eqx.is_array))) / GIB
    )
    histories = full_histories(cfg.total_steps, cfg.log_interval)

    # the async save alone, then waited out: what it blocks on and the whole write; then a cold
    # restore, the files dropped from this node's cache as a requeue on a fresh node finds them
    record["checkpoint"] = []
    for target in targets:
        run_dir = Path(target) / f"bench-checkpoint-{cfg.name}-{os.getpid()}"
        run_dir.mkdir(parents=True, exist_ok=True)
        saves, writes = [], []
        with checkpoint_manager(run_dir) as checkpoints:
            for step in range(1, repeats + 2):
                start = time.perf_counter()
                save_checkpoint(checkpoints, state, step, histories)
                saves.append(time.perf_counter() - start)
                checkpoints.wait_until_finished()
                writes.append(time.perf_counter() - start)
            latest = run_dir / "checkpoints" / str(checkpoints.latest_step())
            files = {
                item.name: sum(f.stat().st_size for f in item.rglob("*") if f.is_file())
                for item in latest.iterdir()
            }
        evict(run_dir)
        with checkpoint_manager(run_dir) as checkpoints:
            restore_s = wall(
                lambda: restore_checkpoint(checkpoints, state, tuple(histories))[0]
            )[1]
        shutil.rmtree(run_dir)
        record["checkpoint"].append(
            dict(
                target=str(Path(target).resolve()),
                save_s=saves,
                write_s=writes,
                restore_s=restore_s,
                file_bytes=files,
            )
        )

    # history.npz at the run's last epoch, its largest, for each log interval
    history_dir = Path(targets[0]) / f"bench-history-{cfg.name}-{os.getpid()}"
    history_dir.mkdir(parents=True, exist_ok=True)
    record["history_s"] = {
        str(interval): min(
            wall(
                write_history,
                history_dir,
                full_histories(cfg.total_steps, interval),
                interval,
            )[1]
            for _ in range(repeats)
        )
        for interval in history_intervals
    }
    shutil.rmtree(history_dir)

    # the eval-catalog diagnostics and the fused epoch, compiled ahead (a restart's compiles, and the
    # memory plans), then a first call of each
    dynamic, static = eqx.partition(state, eqx.is_array)
    catalog = eval_catalog(problem, cfg.seed)
    key_catalog = jr.split(jr.key(cfg.seed), 3)[0]

    def diag(dynamic):
        return diagnostics(
            problem, eqx.combine(dynamic, static), key_catalog, catalog
        )

    def epoch(dynamic):
        new_state, history = train_epoch(
            problem,
            tx,
            eqx.combine(dynamic, static),
            cfg.batch_size,
            steps,
            cfg.draws,
            cfg.micro_batches,
        )
        return eqx.filter(new_state, eqx.is_array), history

    plans, compiled = [], {}
    for stage_name, fn in (("diagnostics", diag), ("epoch", epoch)):
        start = time.perf_counter()
        compiled[stage_name] = stage(
            stage_name, lambda: jax.jit(fn).lower(dynamic).compile()
        )
        record[f"{stage_name}_compile_s"] = time.perf_counter() - start
        plans.append(plan_gib(compiled[stage_name]))
        record[f"{stage_name}_first_s"] = stage(
            stage_name, lambda: wall(compiled[stage_name], dynamic)
        )[1]
    record |= dict(steps=steps, plan_gib=max(plans))

    # train.main's epoch loop, part by part, to the repo's outputs/: `calls` fused epochs of
    # `steps` make one epoch: one call, then enough calls to last twice the background write
    def loop(calls: int, dynamic):
        run_dir = Path(targets[0]) / f"bench-checkpoint-{cfg.name}-{os.getpid()}"
        run_dir.mkdir(parents=True, exist_ok=True)
        parts = dict(epoch_s=[], diagnostics_s=[], save_s=[], history_s=[])
        with checkpoint_manager(run_dir) as checkpoints:
            for i in range(loop_epochs):
                start = time.perf_counter()
                for _ in range(calls):
                    dynamic, losses = compiled["epoch"](dynamic)
                jax.device_get(losses)
                trained = time.perf_counter()
                jax.device_get(compiled["diagnostics"](dynamic))
                diagnosed = time.perf_counter()
                save_checkpoint(
                    checkpoints, eqx.combine(dynamic, static), i + 1, histories
                )
                saved = time.perf_counter()
                write_history(run_dir, histories, cfg.log_interval)
                parts["epoch_s"].append(trained - start)
                parts["diagnostics_s"].append(diagnosed - trained)
                parts["save_s"].append(saved - diagnosed)
                parts["history_s"].append(time.perf_counter() - saved)
        shutil.rmtree(run_dir)
        return dynamic, dict(calls=calls, steps=calls * steps) | parts

    # the step from each loop's first epoch, which has no write in flight
    dynamic, short = stage("epoch", lambda: loop(1, dynamic))
    record["loop"] = [short]
    write_s = float(np.median(record["checkpoint"][0]["write_s"]))
    calls = math.ceil(2 * write_s / short["epoch_s"][0])
    if calls > 1:
        record["loop"].append(stage("epoch", lambda: loop(calls, dynamic))[1])
    first = record["loop"][-1]
    record["step_s"] = first["epoch_s"][0] / first["steps"]
    record["diagnostics_s"] = min(short["diagnostics_s"])
    return record


def main(
    out: str,
    configs: list[str],
    targets: list[str],
    steps: int,
    repeats: int,
    loop_epochs: int,
    history_intervals: list[int],
) -> None:
    device = jax.devices()[0]
    shared = dict(
        job=os.environ.get("SLURM_JOB_ID"),
        chip=device.device_kind,
        cpus=len(os.sched_getaffinity(0)),
        uv_s=float(os.environ.get("BENCH_UV_S", "nan")),
        import_s=time.time() - T0,
        git_s=git_s(),
    )
    print(f"{device.device_kind}, imports {shared['import_s']:.1f} s", flush=True)

    # what this job recorded before a requeue
    done = set()
    if Path(out).exists() and shared["job"]:
        records = map(json.loads, Path(out).read_text().splitlines())
        done = {r["config"] for r in records if r.get("job") == shared["job"]}

    for name in configs:
        if name in done:
            print(f"=== {name}: recorded", flush=True)
            continue
        print(f"=== {name}", flush=True)
        record = dict(shared)
        try:
            record = bench(
                name, record, targets, steps, repeats, loop_epochs, history_intervals
            )
        except OutOfMemory:
            pass
        with open(out, "a") as fh:
            fh.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        jax.clear_caches()
        gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("out")
    parser.add_argument(
        "--configs",
        nargs="+",
        default=[
            f"{b}-{s}" for b in ("narrow", "full") for s in ("XS", "S", "B", "L", "XL")
        ],
    )
    parser.add_argument("--targets", nargs="+", default=["outputs"])
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--loop-epochs", type=int, default=3)
    parser.add_argument("--history-intervals", type=int, nargs="+", default=[100, 1000])
    args = parser.parse_args()
    main(
        args.out,
        args.configs,
        args.targets,
        args.steps,
        args.repeats,
        args.loop_epochs,
        args.history_intervals,
    )
