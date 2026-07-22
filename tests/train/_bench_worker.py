"""Single (config, batch, phase) throughput + peak-memory probe, run as a subprocess.

Reads CFG_NAME, BATCH_SIZE, PHASE (train|eval|gen) from the environment and prints
one JSON line of measurements. Isolated per process for a clean peak-memory read.
"""

import json
import os
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from flax import nnx

import canna  # noqa: F401
from canna.train import TrainState, sample_batch

from _bench import (
    CONFIG_BY_NAME,
    build_model,
    reference_problem,
    synthetic_batch,
)

ODE_STEPS = int(os.environ.get("ODE_STEPS", "4"))
N_TIMED = 10


def peak_mb() -> float:
    try:
        return jax.devices()[0].memory_stats()["peak_bytes_in_use"] / 1e6
    except Exception:
        return float("nan")


def time_ms(fn, n: int = N_TIMED) -> tuple[float, float]:
    t0 = time.monotonic()
    jax.block_until_ready(fn())
    compile_ms = (time.monotonic() - t0) * 1e3
    t0 = time.monotonic()
    for _ in range(n):
        out = fn()
    jax.block_until_ready(out)
    step_ms = (time.monotonic() - t0) / n * 1e3
    return compile_ms, step_ms


def _state(cfg, problem) -> TrainState:
    flow = build_model(cfg, problem)
    optimizer = nnx.Optimizer(
        model=flow,
        tx=optax.chain(
            optax.clip_by_global_norm(1.0), optax.adamw(1e-4, weight_decay=1e-5)
        ),
        wrt=nnx.Param,
    )
    return TrainState(
        problem=problem,
        flow=flow,
        optimizer=optimizer,
        flow_metrics=nnx.metrics.Welford(),
        x_metrics=nnx.metrics.Welford(),
        y_metrics=nnx.metrics.Welford(),
        rngs=nnx.Rngs(0),
    )


def train_probe(cfg, problem, batch):
    state = _state(cfg, problem)
    data = synthetic_batch(problem, batch)
    weights = jnp.ones(3)
    return lambda: state.train_step(data, weights)


def gen_probe(cfg, problem, batch):
    rngs = nnx.Rngs(0)
    return lambda: sample_batch(problem, rngs, batch)


def eval_probe(cfg, problem, batch):
    from canna.eval import sample_posterior

    flow = build_model(cfg, problem)
    sample = problem.train_sample(jr.key(0))
    u0 = jr.normal(jr.key(1), (batch, *sample.xt.shape))
    y0 = jr.normal(jr.key(2), sample.y.shape)
    geometry = problem.geometry
    return lambda: sample_posterior(geometry, flow, u0, y0, ODE_STEPS)


PROBES = {"train": train_probe, "eval": eval_probe, "gen": gen_probe}


def main():
    cfg = CONFIG_BY_NAME[os.environ["CFG_NAME"]]
    batch = int(os.environ["BATCH_SIZE"])
    phase = os.environ["PHASE"]
    problem = reference_problem()

    fn = PROBES[phase](cfg, problem, batch)
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
