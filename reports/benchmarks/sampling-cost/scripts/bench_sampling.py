"""Wall time of the flow's posterior sampler per window, for every band x size at its run config.

A fresh network (the weights do not change the cost), one observation per repeat. One JSON line per
(config, draws) to the output file, skipped when already there.

Run from the repo root (see sampling.sbatch):
    .venv/bin/python reports/benchmarks/sampling-cost/scripts/bench_sampling.py out.jsonl --configs narrow-XS full-B
"""

import argparse
import gc
import json
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.random as jr

from canna.lisa import config
from canna.lisa.sample import sample_posterior
from canna.lisa.train import setup, train_sample


def wall(fn, *args) -> float:
    start = time.perf_counter()
    jax.block_until_ready(fn(*args))
    return time.perf_counter() - start


def bench(name: str, n_draws: int, repeats: int) -> dict:
    """First call (compile included) and the steady per-window time of n_draws posterior draws."""
    cfg = config.load(name)
    problem, _, state = setup(cfg)
    flow = state.flow
    run = lambda z, y, f: sample_posterior(problem, flow, z, y, f, cfg.ode_steps)

    def window(key):
        key_obs, key_z = jr.split(key)
        obs = train_sample(problem, key_obs)
        z = jr.normal(key_z, (n_draws, problem.n_sources, 8))
        return z, obs.y, obs.f

    keys = jr.split(jr.key(0), repeats + 1)
    first_s = wall(run, *window(keys[0]))
    seconds = [wall(run, *window(key)) for key in keys[1:]]
    return dict(
        config=cfg.name,
        chip=jax.devices()[0].device_kind,
        bf16=cfg.bf16,
        draws=n_draws,
        ode_steps=cfg.ode_steps,
        params=sum(
            x.size for x in jax.tree.leaves(eqx.filter(flow, eqx.is_inexact_array))
        ),
        first_s=first_s,
        seconds=seconds,
        best_s=min(seconds),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--draws", nargs="+", type=int, default=[256, 1024])
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    done = set()
    if args.out.exists():
        done = {(r["config"], r["draws"]) for r in map(json.loads, args.out.open())}
    for name in args.configs:
        for n_draws in args.draws:
            if (name, n_draws) in done:
                continue
            record = bench(name, n_draws, args.repeats)
            print(json.dumps(record), flush=True)
            with args.out.open("a") as out:
                out.write(json.dumps(record) + "\n")
            jax.clear_caches()
            gc.collect()


if __name__ == "__main__":
    main()
