"""eval.py's numerics: the Fisher forecast across 45 decades of scale, and RK4 transport."""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from canna.lisa.eval import fisher_draws, sample_posterior, scaled_inverse

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


def unrolled_rk4(problem, flow, u, y, f, ode_steps):
    """The python-loop RK4 that sample_posterior used before moving to fori_loop."""

    def push(u):
        dt = jnp.asarray(1.0 / ode_steps, u.dtype)
        for i in range(ode_steps):
            t = i * dt
            k1 = flow(u, t, y, f)[0]
            k2 = flow(u + k1 * dt / 2, t + dt / 2, y, f)[0]
            k3 = flow(u + k2 * dt / 2, t + dt / 2, y, f)[0]
            k4 = flow(u + k3 * dt, t + dt, y, f)[0]
            u = problem.exp_map(u, (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6)
        return u

    return jax.vmap(push)(u)


@pytest.mark.parametrize("ode_steps", [1, 3])
def test_fori_loop_transport_matches_the_unrolled_rk4(small_flow, ode_steps):
    problem, flow, u0, y, f = small_flow
    post = sample_posterior(problem, flow, u0, y, f, ode_steps)
    assert post.shape == u0.shape and jnp.all(jnp.isfinite(post))
    assert not jnp.allclose(post, u0)  # the perturbed field moves the draws
    assert jnp.allclose(post, unrolled_rk4(problem, flow, u0, y, f, ode_steps), atol=1e-5)


class ConstantVelocity(eqx.Module):
    """A stand-in flow whose velocity along the path is a fixed tangent."""

    v: jax.Array

    def __call__(self, x, t, y, f):
        return jnp.broadcast_to(self.v, x.shape).astype(x.dtype), x, y


@pytest.mark.parametrize("time_power", [1.0, 2.0, 3.0])
def test_the_warped_clock_still_covers_the_whole_path(small_flow, time_power):
    """With ds/dt folded into the ODE, a constant path velocity must move every draw by
    exactly that velocity, whatever the warp: RK4 integrates ds/dt = p (1 - t)^(p-1),
    a polynomial of degree <= 2 here, exactly."""
    problem, _, u0, y, f = small_flow
    # column 1 is log chirp mass, the one Euclidean coordinate: no clipping, no sphere
    v = jnp.zeros(u0.shape[-1]).at[1].set(0.7)
    post = sample_posterior(problem, ConstantVelocity(v), u0, y, f, 8, time_power)
    assert jnp.allclose(post[..., 1], u0[..., 1] + 0.7, atol=1e-10)
    assert jnp.allclose(post[..., 0], u0[..., 0])
