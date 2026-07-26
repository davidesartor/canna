"""Shared config ladder + builders for the size / throughput benchmarks (GPU)."""

import argparse
import csv
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import yaml
from flax import nnx, serialization

import canna
from canna import networks
from canna.problems import LisaGB

BATCH_SIZES = [64, 128, 256]
PEAK_BUDGET_MB = 40_000
TESTS_DIR = os.path.dirname(__file__)
WORKER = os.path.join(TESTS_DIR, "_bench_worker.py")
OUTPUT_DIR = os.path.join("outputs", "bench")
CONFIG_ROOT = Path(canna.__file__).parent / "configs"
CSV_NAME = (
    "{device}_stage_bench.csv"  # per device: jobs on different GPUs run concurrently
)

# the stages one training epoch is made of, in the order train.py runs them
STAGES = ["gen", "metrics", "train", "ckpt", "plot"]
RUN_CONFIGS = ["NoisyPoint-MLP-B", "NoisySinusoid-MMDiT-B"]

OOM_MARKERS = ("RESOURCE_EXHAUSTED", "out of memory", "Out of memory", "OutOfMemory")

# t_obs kept above response_points / 2 / f0_min so LisaGB's band guard passes
BENCH_PROBLEM = dict(n_sources=3, t_obs=1_500_000.0, wdm_freq_bands=128)


@dataclass(frozen=True)
class Config:
    name: str
    hidden_dim: int
    num_blocks: int
    num_heads: int


CONFIGS = [
    Config("S", 384, 8, 6),
    Config("B", 512, 8, 8),
    Config("L", 768, 12, 12),
    Config("XL", 1024, 16, 16),
]

CONFIG_BY_NAME = {c.name: c for c in CONFIGS}


def reference_problem() -> LisaGB:
    return LisaGB(**BENCH_PROBLEM)


def build_model(cfg: Config, problem: LisaGB, seed: int = 0) -> networks.MMDiTFlow:
    sample = problem.train_sample(jr.key(0))
    return networks.MMDiTFlow(
        x_shape=sample.xt.shape,
        y_shape=sample.y.shape,
        hidden_dim=cfg.hidden_dim,
        num_heads=cfg.num_heads,
        num_blocks=cfg.num_blocks,
        rngs=nnx.Rngs(seed),
    )


def param_count(model) -> int:
    state = nnx.to_pure_dict(nnx.state(model, nnx.Param))
    return sum(int(x.size) for x in jax.tree.leaves(state))


def checkpoint_bytes(model) -> int:
    return len(serialization.to_bytes(nnx.to_pure_dict(nnx.state(model))))


def save_text(name: str, text: str):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, f"{name}.txt"), "w") as f:
        f.write(text + "\n")
    print(text, flush=True)


def _spawn(cfg_name: str, batch: int, phase: str, ode_steps: int | None):
    env = {**os.environ, "CFG_NAME": cfg_name, "BATCH_SIZE": str(batch), "PHASE": phase}
    if ode_steps is not None:
        env["ODE_STEPS"] = str(ode_steps)
    return subprocess.run(
        [sys.executable, WORKER], cwd=TESTS_DIR, env=env, capture_output=True, text=True
    )


def run_worker(
    cfg_name: str, batch: int, phase: str, ode_steps: int | None = None
) -> dict:
    proc = _spawn(cfg_name, batch, phase, ode_steps)
    assert (
        proc.returncode == 0
    ), f"{cfg_name}/batch={batch}/{phase} worker failed:\n{proc.stderr[-2000:]}"
    return json.loads(proc.stdout.strip().splitlines()[-1])


def try_worker(
    cfg_name: str, batch: int, phase: str, ode_steps: int | None = None
) -> dict | None:
    proc = _spawn(cfg_name, batch, phase, ode_steps)
    if proc.returncode == 0:
        return json.loads(proc.stdout.strip().splitlines()[-1])
    if proc.returncode == -9 or any(m in proc.stderr for m in OOM_MARKERS):
        return None
    raise AssertionError(
        f"{cfg_name}/batch={batch}/{phase} worker failed (non-OOM):\n{proc.stderr[-2000:]}"
    )


def run_config_args(name: str, batch: int | None = None) -> argparse.Namespace:
    """Rebuild train.py's argparse.Namespace from a run config, as TrainState wants it."""
    defaults = dict(
        seed=0,
        dtype="bfloat16",
        muon=True,
        learning_rate=1e-4,
        weight_decay=0.0,
        batch_size=256,
        total_steps=500_000,
        log_interval=100,
        warmup_frac=0.5,
    )
    with open(CONFIG_ROOT / f"{name}.yaml") as f:
        defaults.update(yaml.safe_load(f))
    if batch is not None:
        defaults["batch_size"] = batch
    return argparse.Namespace(**defaults)


def device_kind() -> str:
    try:
        return jax.devices()[0].device_kind.replace(" ", "_")
    except Exception:
        return "unknown"


def run_stage_worker(run_config: str, batch: int, stage: str) -> dict:
    """Time one epoch stage of one run config, in its own process."""
    env = {
        **os.environ,
        "RUN_CONFIG": run_config,
        "BATCH_SIZE": str(batch),
        "PHASE": stage,
        "MPLBACKEND": "Agg",
    }
    proc = subprocess.run(
        [sys.executable, WORKER], cwd=TESTS_DIR, env=env, capture_output=True, text=True
    )
    assert (
        proc.returncode == 0
    ), f"{run_config}/batch={batch}/{stage} worker failed:\n{proc.stderr[-2000:]}"
    return json.loads(proc.stdout.strip().splitlines()[-1])


def append_csv(rows: list[dict]) -> None:
    """Append measurements to this device's csv, so repeated runs accumulate."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, CSV_NAME.format(device=device_kind()))
    fields = ["device", "backend", "config", "batch", "phase", "n_timed"]
    fields += ["compile_ms", "step_ms", "peak_mb"]
    is_new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if is_new:
            writer.writeheader()
        writer.writerows({k: r.get(k, "") for k in fields} for r in rows)


def max_batch(cfg_name: str, phase: str, start: int = 256, cap: int = 1 << 16) -> int:
    if try_worker(cfg_name, start, phase) is not None:
        b = start
        while b * 2 <= cap and try_worker(cfg_name, b * 2, phase) is not None:
            b *= 2
        return b
    b = start // 2
    while b >= 1 and try_worker(cfg_name, b, phase) is None:
        b //= 2
    return b
