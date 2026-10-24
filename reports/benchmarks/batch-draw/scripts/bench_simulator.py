"""How fast the LISA simulator makes data: jaxgb responses, whitened observations, training batches.

Each size is one jitted, vmapped call, timed as the best of 5 after a warm-up (compile) call; a
size that runs out of memory, or whose call takes over 20 s, ends that sweep. Observations and
training batches are swept at 1, 4 and 16 sources per observation. The JSON is rewritten after
every record, so a job killed mid-sweep keeps what it measured.

Run from the repo root:
    uv run python reports/benchmarks/batch-draw/scripts/bench_simulator.py reports/benchmarks/batch-draw/data/<tag>.json [sizes ...]
"""

import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
import jax
import jax.random as jr
from einops import rearrange

from canna.lisa import config
from canna.lisa.problem import LisaGB, sample_f
from canna.lisa.simulator import (
    preprocess,
    sample_observation,
    sample_physical,
    source_segments,
)
from canna.lisa.train import train_sample

SOURCES = (1, 4, 16)
SLOW = 20.0
RECORDS = []


def timed(fn, *args, repeats: int = 5) -> tuple[float, float]:
    """Best wall time of fn(*args) over `repeats` calls (one when slow), and the warm-up call's excess (compile)."""
    start = time.perf_counter()
    jax.block_until_ready(fn(*args))
    first = time.perf_counter() - start
    best = float("inf")
    for _ in range(repeats if first < SLOW / 4 else 1):
        start = time.perf_counter()
        jax.block_until_ready(fn(*args))
        best = min(best, time.perf_counter() - start)
    return best, max(first - best, 0.0)


def observation(problem, key):
    key_f, key_p, key_o = jr.split(key, 3)
    f = sample_f(problem, key_f)
    p = sample_physical(problem, key_p, f)
    return preprocess(problem, sample_observation(problem, key_o, p, f), f)


def training_batch(problem, cfg, observations, key):
    """train_epoch's batch draw: `observations` windows, cfg.draws stratified (x0, t) draws each."""
    samples = observations * cfg.draws
    key_batch, key_perm, key_u = jr.split(key, 3)
    u = (jr.permutation(key_perm, samples) + jr.uniform(key_u, (samples,))) / samples
    sample = lambda k, uk: train_sample(
        problem,
        k,
        uk,
        time_power=cfg.time_power,
        late_time_mix=cfg.late_time_mix,
        late_time_floor=cfg.late_time_floor,
    )
    return jax.vmap(sample)(
        jr.split(key_batch, observations), rearrange(u, "(b k) -> b k", k=cfg.draws)
    )


def report(out, header, what, n, unit, best, compile_s, **fields) -> bool:
    """Record, save and print one measurement; False when the call was too slow to go on."""
    RECORDS.append(
        dict(
            **fields, n=n, ms=1e3 * best, rate=n / best, unit=unit, compile_s=compile_s
        )
    )
    with open(out, "w") as fh:
        json.dump(header | dict(records=RECORDS), fh, indent=1)
    print(
        f"  {what:<34} n={n:>7}  {1e3 * best:9.2f} ms  {n / best:12.0f} {unit}/s  (compile {compile_s:5.1f} s)",
        flush=True,
    )
    return best < SLOW


def sweep(out, header, what, sizes, build, unit, **fields) -> None:
    """Time build(n)'s call at each size until one runs out of memory or is too slow."""
    for n in sizes:
        try:
            fn, args, count = build(n)
            if not report(out, header, what, count, unit, *timed(fn, *args), **fields):
                return
        except Exception as e:
            print(f"  {what} n={n}: {type(e).__name__}", flush=True)
            return


def main(out: str, sizes: list[int]) -> None:
    device = jax.devices()[0]
    label = (
        f"GPU ({device.device_kind})"
        if device.platform == "gpu"
        else f"CPU ({os.environ.get('SLURM_CPUS_PER_TASK', '?')} cores)"
    )
    header = dict(
        device=label,
        platform=device.platform,
        job=os.environ.get("SLURM_JOB_ID", ""),
        xla_flags=os.environ.get("XLA_FLAGS", ""),
    )
    print(f"device: {label}  x64: {jax.config.jax_enable_x64}", flush=True)
    for name in ("narrow-B", "full-B"):
        cfg = config.load(name)
        problem = LisaGB(**cfg.problem)
        print(
            f"\n{name}: response_points={problem.response_points} window_bins={problem.window_bins}"
            f" draws={cfg.draws}",
            flush=True,
        )

        # jaxgb's response as the trainer calls it: source_segments over n // n_sources windows
        def responses(n):
            windows = n // problem.n_sources
            f = jax.vmap(sample_f, in_axes=(None, 0))(
                problem, jr.split(jr.key(0), windows)
            )
            p = jax.vmap(sample_physical, in_axes=(None, 0, 0))(
                problem, jr.split(jr.key(1), windows), f
            )
            segments = jax.jit(jax.vmap(lambda p, f: source_segments(problem, p, f)))
            return segments, (p, f), windows * problem.n_sources

        sweep(
            out,
            header,
            "get_tdi (sources)",
            sizes,
            responses,
            "src",
            band=name,
            kind="get_tdi",
            sources=None,
        )

        for sources in SOURCES:
            problem = LisaGB(**cfg.problem | dict(n_sources=sources))
            print(f"\n{name}, {sources} sources", flush=True)
            obs = lambda n: (
                jax.jit(
                    lambda k: jax.vmap(observation, in_axes=(None, 0))(
                        problem, jr.split(k, n)
                    )
                ),
                (jr.key(2),),
                n,
            )
            sweep(
                out,
                header,
                "observation (y)",
                sizes,
                obs,
                "obs",
                band=name,
                kind="observation",
                sources=sources,
            )
            batch = lambda n: (
                jax.jit(lambda k: training_batch(problem, cfg, n, k)),
                (jr.key(3),),
                n,
            )
            sweep(
                out,
                header,
                f"train_sample batch (n x {cfg.draws})",
                sizes,
                batch,
                "obs",
                band=name,
                kind="train_sample",
                sources=sources,
            )


if __name__ == "__main__":
    main(
        sys.argv[1],
        [int(s) for s in sys.argv[2:]] or [16, 64, 256, 1024, 4096, 16384, 65536],
    )
