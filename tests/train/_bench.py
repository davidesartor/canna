"""Shared config ladder + builders for the size / throughput benchmarks (GPU)."""

import json
import os
import subprocess
import sys
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax.random as jr
from flax import nnx, serialization

from canna import networks
from canna.problems import LisaGB, TrainSample

BATCH_SIZES = [64, 128, 256]
PEAK_BUDGET_MB = 40_000
TESTS_DIR = os.path.dirname(__file__)
WORKER = os.path.join(TESTS_DIR, "_bench_worker.py")
OUTPUT_DIR = os.path.join("outputs", "bench")

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


def synthetic_batch(problem: LisaGB, batch: int) -> TrainSample:
    sample = problem.train_sample(jr.key(0))
    keys = jr.split(jr.key(1), 6)
    shape = lambda field: (batch, *field.shape)
    return TrainSample(
        xt=jr.normal(keys[0], shape(sample.xt)),
        dx=jr.normal(keys[1], shape(sample.dx)),
        t=jr.uniform(keys[2], (batch,)),
        y=jr.normal(keys[3], shape(sample.y)),
        x_target=jr.normal(keys[4], shape(sample.x_target)),
        y_target=jr.normal(keys[5], shape(sample.y_target)),
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
