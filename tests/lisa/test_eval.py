"""eval.py's numerics: the Fisher forecast across 45 decades of scale, and the sampler."""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from canna.lisa.injections import fisher_draws, scaled_inverse
from canna.lisa.network import LisaFlow
from canna.lisa.sample import sample_posterior

# the physical parameters run from O(1) angles down to ~1e-21 amplitudes
SCALES = jnp.logspace(-22.0, 1.0, 8)


def correlation(key, n=8):
    """A well-conditioned random correlation matrix: all the ill-conditioning is in SCALES."""
    a = jr.normal(key, (n, n))
    c = a @ a.T + n * jnp.eye(n)
    d = jnp.sqrt(jnp.diag(c))
    return c / jnp.outer(d, d)


def test_scaled_inverse_survives_a_45_decade_diagonal():
    c = correlation(jr.key(0))
    cov = c * jnp.outer(SCALES, SCALES)
    # back in the unit-diagonal frame, the inverse must be the correlation's own
    prec = scaled_inverse(cov) * jnp.outer(SCALES, SCALES)
    assert jnp.allclose(prec, jnp.linalg.inv(c), rtol=1e-8, atol=1e-8)


def test_fisher_draws_carry_the_covariance_across_all_the_scales():
    c = correlation(jr.key(1))
    precision = scaled_inverse(c * jnp.outer(SCALES, SCALES))
    mean = SCALES * 3.0
    draws = fisher_draws(jr.key(2), mean, precision, jnp.zeros_like(precision), 40_000)
    assert jnp.all(jnp.isfinite(draws))

    whitened = (draws - mean) / SCALES
    assert jnp.allclose(jnp.mean(whitened, axis=0), 0.0, atol=0.03)
    assert jnp.allclose(jnp.cov(whitened.T), c, atol=0.03)


def test_an_unconstrained_direction_is_held_at_the_prior():
    """A noise-averaged Hessian can dip below the prior, or go negative, along a
    degeneracy; that direction must come out with the prior's width, and finite."""
    n = 8
    v = jnp.linalg.qr(jr.normal(jr.key(3), (n, n)))[0]
    w = jnp.linspace(1.0, 10.0, n).at[0].set(-0.5)  # one negative mode
    s = jnp.outer(SCALES, SCALES)
    precision = (v * w) @ v.T / s
    prior = 0.25 * jnp.eye(n) / s  # prior precision 0.25 in every scaled direction

    draws = fisher_draws(jr.key(4), jnp.zeros(n), precision, prior, 40_000)
    assert jnp.all(jnp.isfinite(draws))

    along = (draws / SCALES) @ v[:, 0]
    assert jnp.isclose(jnp.var(along), 1 / 0.25, rtol=0.05)
    # the well-constrained directions are left alone
    other = (draws / SCALES) @ v[:, -1]
    assert jnp.isclose(jnp.var(other), 1 / 10.0, rtol=0.05)


def test_chunking_the_draws_changes_nothing(small_flow):
    """sample_posterior pushes the draws through in chunks; each draw's path is its own."""
    _, flow, u0, y, f = small_flow
    whole = sample_posterior(flow, u0, y, f, 3, 3.0, chunk=len(u0))
    chunked = sample_posterior(flow, u0, y, f, 3, 3.0, chunk=2)
    # the network computes in float32, the batch size changes its summation order, and the sub-bin f0 embedding amplifies it
    np.testing.assert_allclose(whole, chunked, atol=5e-2)


def perturbed(module, seed=7, scale=0.05):
    params, static = eqx.partition(module, eqx.is_inexact_array)
    leaves, treedef = jax.tree.flatten(params)
    keys = jr.split(jr.key(seed), len(leaves))
    noised = [p + scale * jr.normal(k, p.shape, p.dtype) for p, k in zip(leaves, keys)]
    return eqx.combine(jax.tree.unflatten(treedef, noised), static)


def test_the_default_flow_samples_float64_draws(small_flow):
    """From float64 base draws."""
    problem, _, u0, y, f = small_flow
    flow = perturbed(LisaFlow(problem, 32, 2, 1, key=jr.key(1)))
    post = sample_posterior(flow, u0, y, f, 3)
    assert post.shape == u0.shape and post.dtype == u0.dtype
    assert jnp.all(jnp.isfinite(post)) and not jnp.allclose(post, u0)
