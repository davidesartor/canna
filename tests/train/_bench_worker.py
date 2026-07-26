"""Single (config, batch, phase) throughput + peak-memory probe, run as a subprocess.

Reads CFG_NAME (or RUN_CONFIG), BATCH_SIZE, PHASE from the environment and prints
one JSON line of measurements. Isolated per process for a clean peak-memory read.
"""

import json
import os
import tempfile
import time
from itertools import count
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
import optax
import orbax.checkpoint as ocp
from flax import nnx

import canna  # noqa: F401
from canna.train import TrainState, sample_batch

from _bench import (
    CONFIG_BY_NAME,
    build_model,
    device_kind,
    reference_problem,
    run_config_args,
    synthetic_batch,
)

ODE_STEPS = int(os.environ.get("ODE_STEPS", "4"))
N_TIMED = 10
# the host-side stages are slow and low-variance, so a few repeats is plenty
N_TIMED_BY_PHASE = {"ckpt": 3, "plot": 3}


def peak_mb() -> float:
    if jax.default_backend() == "cpu":
        return float("nan")
    stats = jax.devices()[0].memory_stats()
    if stats is None or "peak_bytes_in_use" not in stats:
        # the peak is read off the BFC arena, which ALLOCATOR=platform never creates
        raise RuntimeError(
            "device reports no memory stats: unset XLA_PYTHON_CLIENT_ALLOCATOR=platform"
        )
    return stats["peak_bytes_in_use"] / 1e6


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


def stage_gen(state, args):
    return lambda: sample_batch(state.problem, state.rngs, args.batch_size)


def stage_metrics(state, args):
    data = sample_batch(state.problem, state.rngs, args.batch_size)
    return lambda: state.update_metrics(data)


def stage_train(state, args):
    data = sample_batch(state.problem, state.rngs, args.batch_size)
    weights = jnp.ones(3)
    return lambda: state.train_step(data, weights)


def stage_ckpt(state, args):
    epochs = args.total_steps // args.log_interval
    loss_hist = jnp.zeros((epochs, args.log_interval, 3))
    tmp = tempfile.mkdtemp()
    checkpoints = ocp.CheckpointManager(
        Path(tmp).absolute(), options=ocp.CheckpointManagerOptions(max_to_keep=1)
    )
    epoch = count(1)

    # orbax writes in a background thread, so the wait is part of the real cost
    def save_once():
        state.save_to(checkpoints, next(epoch), loss_hist)
        checkpoints.wait_until_finished()

    return save_once


def stage_plot(state, args):
    epochs = args.total_steps // args.log_interval
    loss_hist = np.abs(np.random.default_rng(0).normal(size=(epochs, 100, 3)))
    out_pdf = os.path.join(tempfile.mkdtemp(), "losses.pdf")

    # the same full redraw train.py does at the end of every epoch, at full history
    def redraw():
        xs = np.arange(1, epochs + 1) * args.log_interval
        l_lo, l_med, l_up = np.percentile(loss_hist, [10, 50, 90], axis=1)
        fig, ax = plt.subplots()
        loss_colors = {"flow": "#2a86cf", "x": "#1a9e6a", "y": "#8a4bd0"}
        for i, (name, color) in enumerate(loss_colors.items()):
            ax.loglog(xs, l_med[:, i], label=name, color=color, lw=2)
            ax.fill_between(xs, l_lo[:, i], l_up[:, i], color=color, alpha=0.15)
        ax.set(xlabel="step", ylabel="loss", title="bench")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(frameon=False)
        fig.savefig(out_pdf, bbox_inches="tight")
        plt.close(fig)

    return redraw


STAGE_PROBES = {
    "gen": stage_gen,
    "metrics": stage_metrics,
    "train": stage_train,
    "ckpt": stage_ckpt,
    "plot": stage_plot,
}


def main():
    batch = int(os.environ["BATCH_SIZE"])
    phase = os.environ["PHASE"]
    run_config = os.environ.get("RUN_CONFIG")

    if run_config:
        args = run_config_args(run_config, batch)
        state = TrainState.from_config(args)
        fn, name = STAGE_PROBES[phase](state, args), run_config
    else:
        cfg = CONFIG_BY_NAME[os.environ["CFG_NAME"]]
        fn, name = PROBES[phase](cfg, reference_problem(), batch), cfg.name

    n_timed = N_TIMED_BY_PHASE.get(phase, N_TIMED)
    compile_ms, step_ms = time_ms(fn, n_timed)

    print(
        json.dumps(
            {
                "device": device_kind(),
                "backend": jax.default_backend(),
                "config": name,
                "batch": batch,
                "phase": phase,
                "n_timed": n_timed,
                "compile_ms": compile_ms,
                "step_ms": step_ms,
                "peak_mb": peak_mb(),
            }
        )
    )


if __name__ == "__main__":
    main()
