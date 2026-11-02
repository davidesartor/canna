"""The eval's injections, one per log-uniform window, and their Fisher forecasts."""

from functools import partial

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from einops import rearrange
from jaxtyping import Array

from .problem import LisaGB, window_centre
from .simulator import (
    log_likelihood,
    sample_observation,
    sample_physical,
    sample_window,
    snr,
)

N_POSTERIOR = 1024
N_INJECTIONS = 128
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


def sort_by_f0(p: Array) -> Array:
    """Sources by ascending f0, the slot order training teaches the flow; any leading axes."""
    return jnp.take_along_axis(p, jnp.argsort(p[..., 0], axis=-1)[..., None], axis=-2)


def eval_injections(problem: LisaGB, seed: int):
    """N_INJECTIONS windows log-uniform over the band, lowest f first, one prior injection each, sources by f0.

    Returns (latents [N, S, 8], windows, window SNRs, noise keys, prior precision).
    """
    # the seed's stream 3: training splits it in three, and fold_in(key, i) is split(key, n)[i]
    key_pick, key_window, key_noise, key_prior = jr.split(jr.fold_in(jr.key(seed), 3), 4)
    # one log-uniform draw in each of N_INJECTIONS equal log-f0 bins
    lo, hi = np.log10(problem.f0_range)
    u = (np.arange(N_INJECTIONS) + np.asarray(jr.uniform(key_window, (N_INJECTIONS,)))) / N_INJECTIONS
    windows = jax.vmap(partial(window_centre, problem))(jnp.asarray(10.0 ** (lo + u * (hi - lo))))
    latents = sort_by_f0(
        jax.vmap(partial(sample_physical, problem))(jr.split(key_pick, N_INJECTIONS), windows)
    )
    snrs = np.asarray(jax.lax.map(lambda pc: snr(problem, *pc), (latents, windows)))

    # Gaussian-approx prior precision, in physical units, from plain prior draws
    key_prior_window, key_prior_pick = jr.split(key_prior)
    prior_windows = jax.vmap(partial(sample_window, problem))(jr.split(key_prior_window, N_PRIOR_DRAWS))
    prior_latents = jax.vmap(partial(sample_physical, problem))(
        jr.split(key_prior_pick, N_PRIOR_DRAWS), prior_windows
    )
    prior_prec_p = scaled_inverse(jnp.cov(rearrange(prior_latents, "n s p -> (s p) n")))
    # latents stay on device: clean_signal writes the chirp in with .at[].set()
    return latents, windows, snrs, jr.split(key_noise, N_INJECTIONS), prior_prec_p


def fisher_samples(
    problem: LisaGB, latent: Array, f: Array, key_n: Array, prior_prec_p: Array
) -> np.ndarray:
    """Fisher forecast draws [N_POSTERIOR, S, 8]: the nll Hessian averaged over noise realisations, plus the prior.

    Drawn in log fdot (the Jacobian taken at the truth), so the chirp stays positive.
    """
    p0 = latent.reshape(-1)
    nll = lambda p, o: -log_likelihood(problem, p.reshape(latent.shape), o, f)
    # a stream of its own beside eval's noise (key_n) and base draws (fold_in(key_n, 1))
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
