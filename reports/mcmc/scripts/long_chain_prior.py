"""long_chain.py (reports/mcmc/scripts, cc93fb2) with the faint sources' walkers started from the prior, not the Fisher draws.

The reference starts every walker from the injection's Fisher draws, whose f0 width is ~0.3 bins even at SNR 2; on the
crowded problems (narrow-8, full-16) the SNR < 3 sources' MCMC f0 width is then 0.19x / 0.60x the prior's (f0-ordered),
while the flow's is 0.87x / 0.99x. Here sources with SNR < FAINT start from independent prior draws of the same window
(sample_physical), the rest as in mcmc.initial_points; everything else, sampler, prior, likelihood, thinning, is unchanged.
Usage: as long_chain.py, from ~/canna-oneshot with reports/mcmc/scripts on sys.path; put the traces apart with --root.
"""

import sys
from pathlib import Path

import jax
import jax.random as jr
import numpy as np
from einops import rearrange

sys.path.insert(0, str(Path.cwd() / "reports/mcmc/scripts"))
import canna.lisa.mcmc as M
from canna.lisa import config as run_config
from canna.lisa.problem import LisaGB
from canna.lisa.simulator import sample_physical

FAINT = 7.0
PROBLEM = LisaGB(**run_config.load().problem)
fisher_start = M.initial_points


def initial_points(window, n, key):
    theta = rearrange(fisher_start(window, n, key), "n (s k) -> n s k", k=8)
    faint = np.flatnonzero(np.asarray(window.snr) < FAINT)
    if faint.size:
        prior = np.asarray(jax.vmap(lambda k: sample_physical(PROBLEM, k, window.f))(jr.split(jr.fold_in(key, 7), n)))
        for s in faint:
            theta[:, s] = M.to_sampler(prior[:, s], window.centres[s])
        pad = 1e-9 * (window.upper - window.lower)
        theta = np.clip(theta, window.lower + pad, window.upper - pad)
        ceiling = M.log10_fdot_mean(np.exp(theta[..., 0])) + M.FDOT_SIGMAS * M.FDOT_STD
        theta[..., 1] = np.minimum(theta[..., 1], np.log(10) * ceiling - pad[:, 1])
    print(f"prior start for {faint.size} of {theta.shape[1]} sources (SNR < {FAINT:g})", flush=True)
    return rearrange(theta, "n s k -> n (s k)")


M.initial_points = initial_points
import long_chain  # noqa: E402

sys.exit(long_chain.main())
