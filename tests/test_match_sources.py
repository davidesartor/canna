"""match_sources: brute-force (n<=4) and local-search (n>4) assignment."""

import itertools
import time

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import matplotlib.pyplot as plt

from canna import lisa
from _helpers import save_figure, save_text

D = len(lisa.PARAMETER_NAMES)
DRAWS = 32

pair_cost = lambda x, y: jnp.sum(lisa.logarithmic_map(x, y) ** 2)


def cost_matrix(u0, u1):
    """Pairwise assignment cost ``C[i, j] = ||log(u0_i -> u1_j)||^2``."""
    return jax.vmap(jax.vmap(pair_cost, (None, 0)), (0, None))(u0, u1)


@jax.jit
@jax.vmap
def match_cost(u0, u1):
    """Cost of our fast match_sources assignment."""
    return lisa.match_sources(u0, u1)[1]


@jax.jit
@jax.vmap
def sort_cost(u0, u1):
    """Cost of the sort-only initial guess (pair kth-lowest-SNR u0 with kth-lowest-SNR u1)."""
    o0, o1 = jnp.argsort(u0[:, 2]), jnp.argsort(u1[:, 2])
    return jnp.sum(jax.vmap(pair_cost)(u0[o0], u1[o1]))


@jax.jit
@jax.vmap
def no_match_cost(u0, u1):
    """Cost of leaving the sources unmatched (identity pairing)."""
    return jnp.sum(jax.vmap(pair_cost)(u0, u1))


def brute_optimum(u0, u1):
    """Exhaustive minimum assignment cost, batched over the leading axis."""
    n = u0.shape[-2]
    perms = jnp.array(list(itertools.permutations(range(n))))
    one = lambda a, b: jnp.min(jnp.sum(cost_matrix(a, b)[perms, jnp.arange(n)], axis=-1))
    return jax.vmap(one)(u0, u1)


def hungarian(C):
    """Minimum assignment cost of a square cost matrix (numpy-vectorized Kuhn-Munkres)."""
    C = np.asarray(C, dtype=float)
    n = C.shape[0]
    u, v = np.zeros(n + 1), np.zeros(n + 1)
    p, way = np.zeros(n + 1, dtype=int), np.zeros(n + 1, dtype=int)  # p[j] = row at col j
    for i in range(1, n + 1):
        p[0], j0 = i, 0
        minv, used = np.full(n + 1, np.inf), np.zeros(n + 1, dtype=bool)
        while True:
            used[j0] = True
            unused = ~used[1:]
            cur = C[p[j0] - 1] - u[p[j0]] - v[1:]
            better = unused & (cur < minv[1:])
            minv[1:][better], way[1:][better] = cur[better], j0
            cand = np.where(unused, minv[1:], np.inf)
            j1, delta = int(np.argmin(cand)) + 1, cand.min()
            u[p[used]] += delta
            v[used] -= delta
            minv[~used] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            p[j0], j0 = p[way[j0]], way[j0]
    return sum(C[p[j] - 1, j - 1] for j in range(1, n + 1))


def timed(fn):
    """Run once to warm up (JIT compile), then time a second run; return (mean_result, seconds)."""
    jax.block_until_ready(fn())
    t0 = time.perf_counter()
    out = fn()
    jax.block_until_ready(out)
    return float(np.mean(np.asarray(out))), time.perf_counter() - t0


def random_pairs(n, draws=DRAWS):
    keys = jr.split(jr.key(n), draws)
    u0 = jax.vmap(lambda k: jr.uniform(k, (n, D)))(keys)
    u1 = jax.vmap(lambda k: jr.uniform(jr.fold_in(k, 1), (n, D)))(keys)
    return u0, u1


class TestMatchSources:
    @pytest.mark.parametrize("n", [2, 3, 4, 5, 6])
    def test_hungarian_matches_brute(self, n):
        """Hand-rolled Hungarian equals the exhaustive optimum over 32 random draws."""
        u0, u1 = random_pairs(n)
        opt = brute_optimum(u0, u1)
        for i in range(DRAWS):
            assert jnp.isclose(hungarian(cost_matrix(u0[i], u1[i])), opt[i], atol=1e-9)

    def test_match_vs_hungarian(self):
        """Fast match_sources vs the Hungarian optimum across source counts, cost and wall time."""
        counts = [2**i for i in range(1, 11)]  # 2, 4, ..., 1024
        styles = {"no matching": "s:", "sort only": "^-.", "match_sources": "x--", "Hungarian optimum": "o-"}
        cost = {m: [] for m in styles}
        secs = {m: [] for m in styles}

        for n in counts:
            u0, u1 = random_pairs(n)
            C = [cost_matrix(u0[i], u1[i]) for i in range(DRAWS)]  # precomputed, reused by Hungarian
            runs = {
                "no matching": lambda: no_match_cost(u0, u1),
                "sort only": lambda: sort_cost(u0, u1),
                "match_sources": lambda: match_cost(u0, u1),
                "Hungarian optimum": lambda: np.array([hungarian(C[i]) for i in range(DRAWS)]),
            }
            for m, fn in runs.items():
                c, t = timed(fn)
                cost[m].append(c)
                secs[m].append(t)

        fig, (ax, axt) = plt.subplots(2, 1, sharex=True, figsize=(6, 8))
        for m in styles:
            ax.plot(counts, cost[m], styles[m], label=m)
            axt.plot(counts, secs[m], styles[m], label=m)
        for a in (ax, axt):
            a.set_xscale("log", base=2)
            a.set_yscale("log")  # base 10
        ax.set_ylabel(f"mean assignment cost ({DRAWS} draws)")
        axt.set_ylabel(f"wall time (s, {DRAWS} draws)")
        axt.set_xlabel("number of sources")
        ax.legend()
        save_figure(fig, "match_sources_vs_n")

        head = f"{'n':>6} " + " ".join(f"{m:>18}" for m in styles)
        rows = [f"cost ({DRAWS} draws):", head]
        rows += [f"{n:>6} " + " ".join(f"{cost[m][k]:>18.6g}" for m in styles) for k, n in enumerate(counts)]
        rows += ["", f"wall time s ({DRAWS} draws):", head]
        rows += [f"{n:>6} " + " ".join(f"{secs[m][k]:>18.4g}" for m in styles) for k, n in enumerate(counts)]
        save_text("match_sources_vs_n", "\n".join(rows))

        got, opt, sort = (np.array(cost[m]) for m in ("match_sources", "Hungarian optimum", "sort only"))
        assert np.all(got >= opt - 1e-6)  # never beats the optimum
        assert np.all(got <= sort + 1e-6)  # swaps never worsen the sort-only guess
