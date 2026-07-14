"""Shared model-config sweep + builders for the size / throughput benchmarks."""

import functools
import json
import os
import subprocess
import sys
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import nnx
from flax import serialization

from canna import lisa, networks

BATCH_SIZES = [64, 128, 256]
PEAK_BUDGET_MB = (
    40_000  # per-GPU VRAM ceiling; tighten to ~1.2x baseline after first run
)
TESTS_DIR = os.path.dirname(__file__)
WORKER = os.path.join(TESTS_DIR, "_bench_worker.py")

# GPU RESOURCE_EXHAUSTED, or host-RAM OOM kill (SIGKILL) -> treat batch as "does not fit"
OOM_MARKERS = ("RESOURCE_EXHAUSTED", "out of memory", "Out of memory", "OutOfMemory")


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
    """Run one (config, batch, phase) probe in a fresh subprocess; return its JSON line."""
    proc = _spawn(cfg_name, batch, phase, ode_steps)
    assert (
        proc.returncode == 0
    ), f"{cfg_name}/batch={batch}/{phase} worker failed:\n{proc.stderr[-2000:]}"
    return json.loads(proc.stdout.strip().splitlines()[-1])


def try_worker(
    cfg_name: str, batch: int, phase: str, ode_steps: int | None = None
) -> dict | None:
    """Like run_worker but return None if the batch does not fit (OOM); re-raise other errors."""
    proc = _spawn(cfg_name, batch, phase, ode_steps)
    if proc.returncode == 0:
        return json.loads(proc.stdout.strip().splitlines()[-1])
    if proc.returncode == -9 or any(m in proc.stderr for m in OOM_MARKERS):
        return None
    raise AssertionError(
        f"{cfg_name}/batch={batch}/{phase} worker failed (non-OOM):\n{proc.stderr[-2000:]}"
    )


def max_batch(cfg_name: str, phase: str, start: int = 256, cap: int = 1 << 16) -> int:
    """Largest power-of-two batch that fits for (config, phase); 0 if not even one sample fits."""
    if try_worker(cfg_name, start, phase) is not None:
        b = start
        while b * 2 <= cap and try_worker(cfg_name, b * 2, phase) is not None:
            b *= 2
        return b

    # start itself did not fit -- halve until something does
    b = start // 2
    while b >= 1 and try_worker(cfg_name, b, phase) is None:
        b //= 2
    return b


@dataclass(frozen=True)
class Config:
    name: str
    hidden_dim: int
    num_blocks: int
    num_heads: int


# depth/width ladder around the train.py default (B), all head_dim = 64
CONFIGS = [
    Config("S", 384, 8, 6),
    Config("B", 512, 8, 8),  # current train.py default
    Config("L", 768, 12, 12),
    Config("XL", 1024, 16, 16),
]

CONFIG_BY_NAME = {c.name: c for c in CONFIGS}


@functools.cache
def wdm_shape() -> tuple[int, int, int]:
    """True (T, F, C) of the network conditioning, from one WDM transform of zeros."""
    mock = jnp.zeros((lisa.N_SAMPLES, len(lisa.CHANNEL_NAMES)))
    return tuple(int(d) for d in lisa.preprocess_datastream(mock).shape)


def build_model(cfg: Config, dtype=jnp.float32, seed: int = 0) -> networks.MMDiT:
    return networks.MMDiT(
        x_dim=len(lisa.PARAMETER_NAMES),
        y_channels=len(lisa.CHANNEL_NAMES),
        hidden_dim=cfg.hidden_dim,
        num_blocks=cfg.num_blocks,
        num_heads=cfg.num_heads,
        dtype=dtype,
        param_dtype=dtype,
        rngs=nnx.Rngs(seed),
    )


def param_count(model) -> int:
    state = nnx.to_pure_dict(nnx.state(model, nnx.Param))
    return sum(int(x.size) for x in jax.tree.leaves(state))


def checkpoint_bytes(model) -> int:
    """Serialized size of a full-state checkpoint, exactly as train.py writes it."""
    return len(serialization.to_bytes(nnx.to_pure_dict(nnx.state(model))))
