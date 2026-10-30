"""The report's long-chain injections, and the joint reference's log L timed on one of them (mcmc.py runs the chains)."""

import argparse
import sys
import time
import warnings
from functools import partial

warnings.filterwarnings("ignore", message=r"Using `field\(init=False\)`")

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from einops import rearrange

from canna.lisa import config as run_config
from canna.lisa.injections import N_PER_WINDOW, eval_injections
from canna.lisa.mcmc import (
    initial_points,
    log_likelihood,
    log_prior,
    make_window,
    to_sampler,
)
from canna.lisa.problem import LisaGB

# run long and traced on the page: eval windows 1, 3, 5, 7 at SNR ranks 7, 5, 3, 1 (loudest to faint)
WINDOWS = [N_PER_WINDOW * w + k for w, k in ((1, 7), (3, 5), (5, 3), (7, 1))]
TEMPS = [1.0, 1.58, 2.51, 3.98, 6.31, 10.0]


def main() -> int:
    parser = argparse.ArgumentParser(prog="joint_mcmc", description=__doc__)
    parser.add_argument(
        "--injection", type=int, default=WINDOWS[0], help="eval injection"
    )
    parser.add_argument("--nwalker", type=int, default=64)
    parser.add_argument("--temps", type=float, nargs="+", default=TEMPS)
    args = parser.parse_args()

    config = run_config.load()
    problem = LisaGB(**config.problem)
    window = make_window(problem, eval_injections(problem, config.seed), args.injection)
    n_sources = window.truth.shape[0]
    n_chains = args.nwalker * len(args.temps)
    print(
        f"{config.name} injection {args.injection}: window SNR {window.window_snr:.1f}, sources SNR"
        f" {np.round(window.snr, 1)}; {n_chains} chains, backend {jax.default_backend()}",
        flush=True,
    )

    # the truth must sit inside the boxes, the start inside the prior
    prior = partial(log_prior, window)
    likelihood = partial(log_likelihood, problem, window)
    truth = rearrange(
        np.stack(
            [to_sampler(window.truth[s], window.centres[s]) for s in range(n_sources)]
        ),
        "s k -> (s k)",
    )
    print(
        f"log L at truth {float(likelihood(jnp.asarray(truth))):.2f},"
        f" log prior at truth {float(prior(jnp.asarray(truth))):.2f}",
        flush=True,
    )
    p0 = jnp.asarray(initial_points(window, n_chains, jr.key(17)))
    assert bool(jnp.all(jnp.isfinite(jax.vmap(prior)(p0)))), "start outside the prior"

    batch = jax.jit(jax.vmap(likelihood))
    batch(p0).block_until_ready()
    start = time.perf_counter()
    for _ in range(20):
        batch(p0).block_until_ready()
    print(
        f"log L over {n_chains} chains: {(time.perf_counter() - start) / 20 * 1e3:.2f} ms"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
