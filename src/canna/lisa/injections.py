"""The eval's ten injections at the SNR deciles of prior draws, and their Fisher forecasts."""

import itertools
from functools import partial

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from einops import rearrange
from jaxtyping import Array

from .problem import LisaGB, f0_window, sample_f
from .simulator import log_likelihood, sample_observation, sample_physical, snr

N_POSTERIOR = 1024
N_CANDIDATES = 1024
N_QUANTILES = 10
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
    """The ten eval injections at the SNR deciles of N_CANDIDATES prior draws.

    Returns (latents [10, S, 8], windows, window SNRs, quantiles, noise keys, prior precision).
    """
    key_pick, key_window, key_noise = jr.split(jr.key(seed), 3)
    windows = jax.vmap(partial(sample_f, problem))(jr.split(key_window, N_CANDIDATES))
    latents = jax.vmap(partial(sample_physical, problem))(
        jr.split(key_pick, N_CANDIDATES), windows
    )
    # Gaussian-approx prior precision, in physical units
    prior_prec_p = scaled_inverse(jnp.cov(rearrange(latents, "n s p -> (s p) n")))
    snrs = np.asarray(jax.lax.map(lambda pc: snr(problem, *pc), (latents, windows)))
    quantiles = np.linspace(1.0 / N_QUANTILES, 1.0, N_QUANTILES)
    chosen = np.argsort(snrs)[np.round(quantiles * (N_CANDIDATES - 1)).astype(int)]
    # latents stay on device: clean_signal writes the chirp in with .at[].set()
    return (
        latents[chosen],
        windows[chosen],
        snrs[chosen],
        quantiles,
        jr.split(key_noise, N_QUANTILES),
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
