"""Time and size one LISA training step at a given batch and micro-batch count; one JSON line per point.

The step is the trainer's fused epoch (batch draw + train_step), compiled ahead for XLA's memory
plan, then best of `repeats` calls after a warm-up. The batch draw is also timed alone, as a scan,
and `prepare` times the schedule-free NorMuon update on fixed gradients once per config, so
fwd+bwd = step - draw - optimizer. The batch counts observations, `draws` (x0, t) samples each.
An out-of-memory point is a record with "oom". sweep_micro.py runs many points in one process.

Run from the repo root:
    LISA_CONFIG=full-B uv run --extra cuda python reports/benchmarks/step-time-memory/scripts/bench_micro.py out.jsonl --batch 256 [--micro 1] [--steps 20]
"""

import argparse
import json
import os
import time
import warnings
from collections.abc import Callable
from functools import partial

warnings.filterwarnings("ignore")
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from einops import rearrange

from canna.lisa import config
from canna.lisa.train import setup, train_epoch, train_sample

GIB = 2**30


def timed(fn, *args, repeats: int) -> float:
    """Best wall time of fn(*args) over `repeats` calls after a warm-up call."""
    jax.block_until_ready(fn(*args))
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        jax.block_until_ready(fn(*args))
        best = min(best, time.perf_counter() - start)
    return best


def plan_gib(compiled) -> float | None:
    """XLA's planned device memory for a compiled call (arguments + outputs + temporaries)."""
    try:
        m = compiled.memory_analysis()
        return (
            m.argument_size_in_bytes
            + m.output_size_in_bytes
            + m.temp_size_in_bytes
            - m.alias_size_in_bytes
        ) / GIB
    except Exception:
        return None


def is_oom(e: Exception) -> bool:
    return "RESOURCE_EXHAUSTED" in str(e) or "out of memory" in str(e).lower()


def prepare(name: str, repeats: int) -> tuple[tuple, dict]:
    """Set up one config, time its optimizer update; (config, problem, tx, state) and its shared record."""
    cfg = config.load(name)
    problem, tx, state = setup(cfg)
    device = jax.devices()[0]
    params = eqx.filter(state.flow, eqx.is_inexact_array)
    param_groups = {}
    for path, leaf in jax.tree_util.tree_leaves_with_path(params):
        name = jax.tree_util.keystr(path)
        group = (
            "embeddings and the rest"
            if not name.startswith(".backbone")
            else (
                "attention"
                if ".attention." in name
                else "adaLN modulation" if ".mod_" in name else "MLPs"
            )
        )
        param_groups[group] = param_groups.get(group, 0) + leaf.size

    grads = jax.tree.map(lambda p: 1e-3 * jnp.ones_like(p), params)

    @jax.jit
    def optimizer(params, opt_state):
        def body(carry, _):
            p, s = carry
            u, s = tx.update(grads, s, p)
            return (optax.apply_updates(p, u), s), None

        return jax.lax.scan(body, (params, opt_state), length=10)[0]

    optimizer_s = timed(optimizer, params, state.opt_state, repeats=repeats) / 10
    record = dict(
        config=cfg.name,
        chip=device.device_kind,
        bf16=cfg.bf16,
        draws=cfg.draws,
        hidden_dim=cfg.network["hidden_dim"],
        num_heads=cfg.network["num_heads"],
        num_blocks=cfg.network["num_blocks"],
        params=sum(x.size for x in jax.tree.leaves(params)),
        param_groups=param_groups,
        optimizer_s=optimizer_s,
        limit_gib=(device.memory_stats() or {}).get("bytes_limit", float("nan")) / GIB,
    )
    return (cfg, problem, tx, state), record


def compile_point(
    prepared: tuple,
    record: dict,
    batch: int,
    micro_batches: int,
    repeats: int,
    steps: int | None = None,
) -> Callable[[], dict]:
    """Compile one point; calling the result times it and gives the record (step and draw time and plan).

    Compiling touches no device memory without autotuning, so sweep_micro.py compiles ahead in threads.
    """
    cfg, problem, tx, state = prepared
    steps = steps or max(2, min(10, 1024 * micro_batches // batch))

    record = record | dict(
        batch=batch,
        micro_batches=micro_batches,
        micro=batch // micro_batches,
        steps=steps,
        repeats=repeats,
    )
    sample = partial(train_sample, problem)

    def draw(key):
        u = jr.uniform(jr.fold_in(key, 1), (batch * cfg.draws,))
        return jax.vmap(sample)(
            jr.split(key, batch), rearrange(u, "(b k) -> b k", k=cfg.draws)
        )

    record["batch_bytes"] = sum(
        x.size * x.dtype.itemsize
        for x in jax.tree.leaves(jax.eval_shape(draw, jr.key(0)))
    )

    # the trainer's fused epoch, compiled ahead so its memory plan is known before it runs
    dynamic, static = eqx.partition(state, eqx.is_array)

    def epoch(dynamic):
        new_state, history = train_epoch(
            problem,
            tx,
            eqx.combine(dynamic, static),
            batch * cfg.draws,
            steps,
            cfg.draws,
            micro_batches,
        )
        return eqx.filter(new_state, eqx.is_array), history

    try:
        start = time.perf_counter()
        compiled = jax.jit(epoch).lower(dynamic).compile()
        record |= dict(
            compile_s=time.perf_counter() - start, plan_gib=plan_gib(compiled)
        )
    except Exception as e:
        if not is_oom(e):
            raise
        oom = record | dict(oom=str(e).split("\n")[0][:300])
        return lambda: oom

    # the batch draw alone, scanned as the trainer runs it
    def draws(key):
        def body(k, _):
            k, kb = jr.split(k)
            b = draw(kb)
            return k, jnp.sum(b.x1) + jnp.sum(b.y)

        return jax.lax.scan(body, key, length=steps)[1]

    try:
        compiled_draws = jax.jit(draws).lower(jr.key(1)).compile()
        record["draw_plan_gib"] = plan_gib(compiled_draws)
    except Exception as e:
        if not is_oom(e):
            raise
        compiled_draws = None

    def measure() -> dict:
        try:
            step_s = timed(compiled, dynamic, repeats=repeats) / steps
        except Exception as e:
            if not is_oom(e):
                raise
            return record | dict(oom=str(e).split("\n")[0][:300])
        draw_s = None
        if compiled_draws is not None:
            try:
                draw_s = timed(compiled_draws, jr.key(1), repeats=repeats) / steps
            except Exception as e:
                if not is_oom(e):
                    raise
        return record | dict(step_s=step_s, draw_s=draw_s)

    return measure


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("out")
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument(
        "--micro",
        type=int,
        default=1,
        help="micro-batch count (gradient accumulation steps)",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--steps", type=int, help="steps per timed epoch")
    cli = parser.parse_args()
    prepared, shared = prepare(os.environ.get("LISA_CONFIG"), cli.repeats)
    record = compile_point(
        prepared, shared, cli.batch, cli.micro, cli.repeats, cli.steps
    )()
    # one point per process, so the allocator's peak is this point's own
    stats = jax.devices()[0].memory_stats() or {}
    record["peak_gib"] = stats.get("peak_bytes_in_use", float("nan")) / GIB
    with open(cli.out, "a") as fh:
        fh.write(json.dumps(record) + "\n")
    print(json.dumps(record), flush=True)
    raise SystemExit(3 if "oom" in record else 0)
