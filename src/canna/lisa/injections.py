"""The eval's grid of injections, windows by frequency and injections by SNR, and their Fisher forecasts."""

import itertools
from functools import partial

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from einops import rearrange
from jaxtyping import Array

from .problem import LisaGB, f0_window
from .simulator import (
    log_likelihood,
    sample_observation,
    sample_physical,
    sample_window,
    snr,
    source_snrs,
)

N_POSTERIOR = 1024
N_WINDOWS = 8
N_PER_WINDOW = 8
N_PRIOR_DRAWS = 1024
N_FISHER_DRAWS = 32


def scaled_inverse(m: Array) -> Array:
    """Inverse of a symmetric matrix, taken in its unit-diagonal frame (the diagonal spans ~45 decades)."""
    d = jnp.sqrt(jnp.abs(jnp.diag(m)))
    return jnp.linalg.inv(m / jnp.outer(d, d)) / jnp.outer(d, d)


def fisher_draws(
    key: Array, mean: Array, precision: Array, prior_precision: Array, n: int
) -> Array:
    """n Gaussian draws with the given precision, each eigendirection held at least at the prior's."""
    precision = (precision + precision.T) / 2
    d = jnp.sqrt(jnp.abs(jnp.diag(precision)))
    w, v = jnp.linalg.eigh(precision / jnp.outer(d, d))
    prior_w = jnp.einsum("ik,ij,jk->k", v, prior_precision / jnp.outer(d, d), v)
    w = jnp.maximum(w, prior_w)
    z = jr.normal(key, (n, mean.size), mean.dtype)
    return mean + (z / jnp.sqrt(w)) @ v.T / d


def sort_by_snr(problem: LisaGB, p: Array, f: Array) -> Array:
    """The window's sources loudest first: only to show them apart, training and sampling never sort."""
    return p[jnp.argsort(-source_snrs(problem, p, f))]


def match_sources(
    problem: LisaGB, draws: np.ndarray, truth: np.ndarray, f
) -> np.ndarray:
    """Relabel each draw's sources onto the true ones: the cheapest permutation in f0, log A, sky."""
    lo, hi = (float(v) for v in f0_window(problem, f))
    perms = np.array(list(itertools.permutations(range(truth.shape[0]))))

    def features(p):
        lam, beta = p[..., 3], p[..., 4]
        return np.stack(
            [
                p[..., 0] / (hi - lo),
                np.log10(np.abs(p[..., 2]) + 1e-300),
                np.cos(beta) * np.cos(lam),
                np.cos(beta) * np.sin(lam),
                np.sin(beta),
            ],
            axis=-1,
        )

    cost = ((features(draws)[:, perms] - features(truth)[None, None]) ** 2).sum(
        (-1, -2)
    )
    best = perms[np.argmin(cost, axis=1)]
    return np.take_along_axis(draws, best[:, :, None], axis=1)


def eval_injections(problem: LisaGB, seed: int):
    """N_WINDOWS windows at the window prior's quantiles, lowest f first, each with N_PER_WINDOW prior injections,
    faintest first; each injection's sources loudest first.

    Returns (latents [W K, S, 8], windows, window SNRs, grid [W K, 2] of (window, SNR rank), noise keys, prior precision).
    """
    # the seed's stream 3: training splits it in three, and fold_in(key, i) is split(key, n)[i]
    key_pick, key_window, key_noise, key_prior = jr.split(jr.fold_in(jr.key(seed), 3), 4)
    sample_windows = lambda key: jax.vmap(partial(sample_window, problem))(
        jr.split(key, N_PRIOR_DRAWS)
    )
    # the midpoints of the window prior's N_WINDOWS quantile bins: the band as training sees it
    pool = np.sort(np.asarray(sample_windows(key_window)))
    levels = (np.arange(N_WINDOWS) + 0.5) / N_WINDOWS
    windows = np.repeat(pool[np.round(levels * (N_PRIOR_DRAWS - 1)).astype(int)], N_PER_WINDOW)
    latents = jax.vmap(partial(sample_physical, problem))(
        jr.split(key_pick, windows.size), jnp.asarray(windows)
    )
    snrs = np.asarray(jax.lax.map(lambda pc: snr(problem, *pc), (latents, jnp.asarray(windows))))
    ranks = np.argsort(rearrange(snrs, "(w k) -> w k", k=N_PER_WINDOW), axis=1)
    order = (ranks + N_PER_WINDOW * np.arange(N_WINDOWS)[:, None]).ravel()
    picked = jax.vmap(partial(sort_by_snr, problem))(latents[order], jnp.asarray(windows))
    grid = np.stack(np.divmod(np.arange(windows.size), N_PER_WINDOW), axis=1)

    # Gaussian-approx prior precision, in physical units, from plain prior draws
    key_prior_window, key_prior_pick = jr.split(key_prior)
    prior_latents = jax.vmap(partial(sample_physical, problem))(
        jr.split(key_prior_pick, N_PRIOR_DRAWS), sample_windows(key_prior_window)
    )
    prior_prec_p = scaled_inverse(jnp.cov(rearrange(prior_latents, "n s p -> (s p) n")))
    # latents stay on device: clean_signal writes the chirp in with .at[].set()
    return (
        picked,
        jnp.asarray(windows),
        snrs[order],
        grid,
        jr.split(key_noise, windows.size),
        prior_prec_p,
    )


def fisher_samples(
    problem: LisaGB, latent: Array, f: Array, key_n: Array, prior_prec_p: Array
) -> np.ndarray:
    """Fisher forecast draws [N_POSTERIOR, S, 8]: the nll Hessian averaged over noise realisations, plus the prior.

    Drawn in log fdot (the Jacobian taken at the truth), so the chirp stays positive.
    """
    p0 = latent.reshape(-1)
    nll = lambda p, o: -log_likelihood(problem, p.reshape(latent.shape), o, f)
    # past split(key_n, N_POSTERIOR), the flow draws' keys in eval
    key_replicas, key_draws = jr.split(jr.fold_in(key_n, N_POSTERIOR), 2)
    replicas = jax.vmap(partial(sample_observation, problem), in_axes=(0, None, None))(
        jr.split(key_replicas, N_FISHER_DRAWS), latent, f
    )
    prec_p = (
        jax.lax.map(lambda o: jax.hessian(nll)(p0, o), replicas).mean(0) + prior_prec_p
    )

    # dp/dq at the truth, q being p with log fdot in place of fdot
    jac = jnp.ones_like(latent).at[..., 1].set(latent[..., 1]).reshape(-1)
    q0 = latent.at[..., 1].set(jnp.log(latent[..., 1])).reshape(-1)
    draws = fisher_draws(
        key_draws,
        q0,
        prec_p * jnp.outer(jac, jac),
        prior_prec_p * jnp.outer(jac, jac),
        N_POSTERIOR,
    ).reshape(N_POSTERIOR, *latent.shape)
    draws = draws.at[..., 1].set(jnp.exp(draws[..., 1]))
    return np.asarray(draws)
