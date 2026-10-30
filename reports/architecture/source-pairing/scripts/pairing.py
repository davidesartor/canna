"""Path length per source under each pairing of a prior draw's sources with the target's, for the source-pairing page.

Run from the repo root (JAX: a job, not the login node):
    uv run --frozen python reports/architecture/source-pairing/scripts/pairing.py reports/architecture/source-pairing/data/pairing.json
PAIRING_WINDOWS shrinks the run for a smoke test.
"""

import itertools
import json
import os
import subprocess
import sys
import time

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from canna.lisa import config
from canna.lisa.chart import physical_to_flow
from canna.lisa.injections import sort_by_snr
from canna.lisa.network import LOSS_GROUPS
from canna.lisa.problem import LisaGB
from canna.lisa.simulator import sample_physical, sample_window

BANDS = ("narrow", "full")
WINDOWS = int(os.environ.get("PAIRING_WINDOWS", 2**16))
CHUNK = 1024
GROUPS = ["f0", "fdot", "sky", "z+", "z-"]
PAIRINGS = ["f0", "nearest", "snr", "random"]


def nearest_permutation(x0: jax.Array, x1: jax.Array) -> jax.Array:
    """x0's sources relabelled onto x1's at the least total squared distance, over all S! labellings."""
    perms = jnp.array(list(itertools.permutations(range(x1.shape[0]))))
    cost = jnp.sum(jnp.square(x0[:, None] - x1[None]), axis=-1)
    total = jnp.sum(cost[perms, jnp.arange(x1.shape[0])], axis=-1)
    return x0[perms[jnp.argmin(total)]]


def f0_rank_pairing(x0: jax.Array, x1: jax.Array) -> jax.Array:
    """x0's sources relabelled so each has the f0 rank of x1's source in its slot (train's sort by f0, both ends)."""
    rank = jnp.argsort(jnp.argsort(x1[:, 0]))
    return x0[jnp.argsort(x0[:, 0])][rank]


def window_paths(problem: LisaGB, key: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Squared path length per (pairing, target source, group), and whether f0 order is the nearest labelling or loudest first."""
    key_f, key_1, key_0, key_r = jr.split(key, 4)
    f = sample_window(problem, key_f)
    # both ends loudest first, so the SNR order is the draws as they stand
    x1, x0 = (
        physical_to_flow(
            problem, sort_by_snr(problem, sample_physical(problem, k, f), f), f
        )
        for k in (key_1, key_0)
    )
    f0_paired, nearest = f0_rank_pairing(x0, x1), nearest_permutation(x0, x1)
    paired = jnp.stack([f0_paired, nearest, x0, jr.permutation(key_r, x0)])
    squared = jnp.square(paired - x1)
    groups = jnp.stack([squared[..., g].sum(-1) for g in LOSS_GROUPS], axis=-1)
    same = jnp.stack([jnp.all(f0_paired == nearest), jnp.all(f0_paired == x0)])
    return groups, same


def measure(band: str) -> dict:
    cfg = config.load(f"{band}-B")
    problem = LisaGB(**cfg.problem)
    run = jax.jit(jax.vmap(lambda k: window_paths(problem, k)))
    keys = jr.split(jr.key(0), WINDOWS)
    t0 = time.time()
    parts = [run(keys[i : i + CHUNK]) for i in range(0, WINDOWS, CHUNK)]
    paths = np.concatenate([np.asarray(p[0], np.float64) for p in parts])
    same = np.concatenate([np.asarray(p[1]) for p in parts])
    print(f"{band}: {WINDOWS} windows in {time.time() - t0:.0f} s", flush=True)

    # per window, the mean over sources: the unit the paper quotes
    per_window = paths.mean(axis=2)
    total = per_window.sum(-1)
    result = dict(
        sources=problem.n_sources,
        windows=WINDOWS,
        mean={k: per_window[:, i].mean(0).tolist() for i, k in enumerate(PAIRINGS)},
        total={k: float(total[:, i].mean()) for i, k in enumerate(PAIRINGS)},
        total_se={
            k: float(total[:, i].std() / np.sqrt(WINDOWS))
            for i, k in enumerate(PAIRINGS)
        },
        # target source s is the s-th loudest
        by_rank={k: paths[:, i].mean(0).tolist() for i, k in enumerate(PAIRINGS)},
        ratio_quantiles={
            k: np.quantile(total[:, i] / total[:, 0], [0.1, 0.5, 0.9]).tolist()
            for i, k in enumerate(PAIRINGS)
        },
        f0_is_nearest=float(same[:, 0].mean()),
        f0_is_snr_order=float(same[:, 1].mean()),
    )
    for k in PAIRINGS:
        print(
            f"  {k:8s} total {result['total'][k]:.3f} ± {result['total_se'][k]:.3f}  "
            + "  ".join(f"{g} {v:.3f}" for g, v in zip(GROUPS, result["mean"][k]))
        )
    print(
        f"  f0 order = nearest labelling in {result['f0_is_nearest']:.1%},"
        f" = loudest first in {result['f0_is_snr_order']:.1%} of windows"
    )
    return result


def main() -> None:
    out = sys.argv[1]
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    data = dict(
        commit=commit,
        groups=GROUPS,
        pairings=PAIRINGS,
        date=time.strftime("%Y-%m-%d"),
        bands={band: measure(band) for band in BANDS},
    )
    with open(out, "w") as file:
        json.dump(data, file, indent=1)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
