"""Single (config, batch, phase) throughput + peak-memory probe, run as a subprocess.

Isolated per process so peak GPU memory is the clean high-water of one phase.
Reads CFG_NAME, BATCH_SIZE, PHASE (train|eval) from the environment and prints
one JSON line of measurements.
"""

import json
import os
import sys
import time

# grow the allocator on demand: peak_bytes_in_use then reads the true high-water,
# and the probe OOMs at the real limit rather than at a preallocated pool
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as jr
import optax
from flax import nnx

from canna import lisa
from _bench import CONFIG_BY_NAME, build_model, wdm_shape

# the real training / eval kernels live in the driver scripts (single source of truth)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from train import TrainBatch, sample_train_batch, train_step

NET_DTYPE = jnp.float32
ODE_STEPS = int(os.environ.get("ODE_STEPS", "4"))
N_TIMED = 10


def peak_mb() -> float:
    try:
        return jax.devices()[0].memory_stats()["peak_bytes_in_use"] / 1e6
    except Exception:
        return float("nan")


def time_ms(fn, n: int = N_TIMED) -> tuple[float, float]:
    """(compile ms, steady per-call ms): first call triggers XLA compile, then time n calls."""
    t0 = time.monotonic()
    jax.block_until_ready(fn())
    compile_ms = (time.monotonic() - t0) * 1e3
    t0 = time.monotonic()
    for _ in range(n):
        out = fn()
    jax.block_until_ready(out)
    step_ms = (time.monotonic() - t0) / n * 1e3
    return compile_ms, step_ms


def train_probe(cfg, batch, T, F, C, S, P):
    """One train_step closure over a synthetic batch, mutating flow/optimizer in place."""
    flow = build_model(cfg, dtype=NET_DTYPE)
    optimizer = nnx.Optimizer(
        flow,
        optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(1e-4, weight_decay=1e-5),
        ),
        wrt=nnx.Param,
    )
    # synthetic batch mirroring train precision: net inputs fp32, loss targets fp64
    k = jr.split(jr.key(0), 6)
    batch_data = TrainBatch(
        ut=jr.normal(k[0], (batch, S, P), NET_DTYPE),
        du=jr.normal(k[1], (batch, S, P)),
        t=jr.uniform(k[2], (batch,), NET_DTYPE),
        y=jr.normal(k[3], (batch, T, F, C), NET_DTYPE),
        u1=jr.uniform(k[4], (batch, S, P)),
        y_clean=jr.normal(k[5], (batch, T, F, C)),
    )
    return lambda: train_step(flow, optimizer, batch_data, aux_weight=1.0)


def gen_probe(cfg, batch, T, F, C, S, P):
    """Time the real jitted physics batch generator (config-independent)."""
    rngs = nnx.Rngs(0)
    return lambda: sample_train_batch(rngs, batch)


def eval_probe(cfg, batch, T, F, C, S, P):
    """Draw `batch` posterior samples for a single synthetic observation y."""
    from eval import (
        sample_posterior,
    )  # lazy: eval.py's legacy import shouldn't gate train/gen

    flow = build_model(cfg, dtype=NET_DTYPE)
    u0 = jr.uniform(jr.key(1), (batch, S, P), NET_DTYPE)
    y0 = jr.normal(jr.key(2), (T, F, C), NET_DTYPE)
    return lambda: sample_posterior(flow, u0, y0, ODE_STEPS)


PROBES = {"train": train_probe, "eval": eval_probe, "gen": gen_probe}


def main():
    cfg = CONFIG_BY_NAME[os.environ["CFG_NAME"]]
    batch = int(os.environ["BATCH_SIZE"])
    phase = os.environ["PHASE"]
    T, F, C = wdm_shape()
    S, P = lisa.N_SOURCES, len(lisa.PARAMETER_NAMES)

    fn = PROBES[phase](cfg, batch, T, F, C, S, P)
    compile_ms, step_ms = time_ms(fn)

    print(
        json.dumps(
            {
                "config": cfg.name,
                "batch": batch,
                "phase": phase,
                "compile_ms": compile_ms,
                "step_ms": step_ms,
                "peak_mb": peak_mb(),
            }
        )
    )


if __name__ == "__main__":
    main()
