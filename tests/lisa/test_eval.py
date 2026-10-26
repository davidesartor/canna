"""eval.py's numerics: the Fisher forecast across 45 decades of scale, and RK4 transport."""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

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


def unrolled_rk4(flow, u, y, f, ode_steps):
    """The python-loop RK4 that sample_posterior used before moving to fori_loop."""

    def push(u):
        dt = jnp.asarray(1.0 / ode_steps, u.dtype)
        for i in range(ode_steps):
            t = i * dt
            k1 = flow(u, t, y, f)
            k2 = flow(u + k1 * dt / 2, t + dt / 2, y, f)
            k3 = flow(u + k2 * dt / 2, t + dt / 2, y, f)
            k4 = flow(u + k3 * dt, t + dt, y, f)
            u = u + (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6
        return u

    return jax.vmap(push)(u)


@pytest.mark.parametrize("ode_steps", [1, 3])
def test_fori_loop_transport_matches_the_unrolled_rk4(small_flow, ode_steps):
    _, flow, u0, y, f = small_flow
    post = sample_posterior(flow, u0, y, f, ode_steps)
    assert post.shape == u0.shape and jnp.all(jnp.isfinite(post))
    assert not jnp.allclose(post, u0)  # the perturbed field moves the draws
    # float32 noise, amplified by the flow's sub-bin f0 embedding, reaches ~1e-2 by three steps
    np.testing.assert_allclose(post, unrolled_rk4(flow, u0, y, f, ode_steps), atol=3e-2)


class ConstantVelocity(eqx.Module):
    """A stand-in flow whose velocity along the path is a fixed tangent."""

    v: jax.Array
    prediction: str = eqx.field(static=True, default="velocity")
    self_condition: bool = eqx.field(static=True, default=False)

    def encode(self, y, f):
        return None

    def __call__(self, x, t, y, f, x_hat=None, cache=None):
        return jnp.broadcast_to(self.v, x.shape).astype(x.dtype)


@pytest.mark.parametrize("time_power", [1.0, 2.0, 3.0])
def test_the_warped_clock_still_covers_the_whole_path(small_flow, time_power):
    """With ds/dt folded into the ODE, a constant path velocity must move every draw by
    exactly that velocity, whatever the warp: RK4 integrates ds/dt = p (1 - t)^(p-1),
    a polynomial of degree <= 2 here, exactly."""
    _, _, u0, y, f = small_flow
    # column 1 is the standardised log fdot
    v = jnp.zeros(u0.shape[-1]).at[1].set(0.7)
    post = sample_posterior(ConstantVelocity(v), u0, y, f, 8, time_power)
    assert jnp.allclose(post[..., 1], u0[..., 1] + 0.7, atol=1e-10)
    assert jnp.allclose(post[..., 0], u0[..., 0])


def test_chunking_the_draws_changes_nothing(small_flow):
    """sample_posterior pushes the draws through in chunks; each draw's path is its own."""
    _, flow, u0, y, f = small_flow
    whole = sample_posterior(flow, u0, y, f, 3, 3.0, chunk=len(u0))
    chunked = sample_posterior(flow, u0, y, f, 3, 3.0, chunk=2)
    # the network computes in float32, the batch size changes its summation order, and the sub-bin f0 embedding amplifies it
    np.testing.assert_allclose(whole, chunked, atol=3e-2)


def perturbed(module, seed=7, scale=0.05):
    params, static = eqx.partition(module, eqx.is_inexact_array)
    leaves, treedef = jax.tree.flatten(params)
    keys = jr.split(jr.key(seed), len(leaves))
    noised = [p + scale * jr.normal(k, p.shape, p.dtype) for p, k in zip(leaves, keys)]
    return eqx.combine(jax.tree.unflatten(treedef, noised), static)


def test_the_default_flow_samples_float64_draws(small_flow):
    """x-prediction, self-conditioned on its float32 x1, from float64 base draws."""
    problem, _, u0, y, f = small_flow
    flow = perturbed(LisaFlow(problem, 32, 2, 1, key=jr.key(1)))
    assert flow.prediction == "x" and flow.self_condition
    post = sample_posterior(flow, u0, y, f, 3)
    assert post.shape == u0.shape and post.dtype == u0.dtype
    assert jnp.all(jnp.isfinite(post)) and not jnp.allclose(post, u0)
