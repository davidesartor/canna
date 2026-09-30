"""eval.py's numerics: the Fisher forecast across 45 decades of scale, and RK4 transport."""

from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest
import yaml

import canna.lisa as lisa
from canna.lisa import LisaFlow, LisaGB, train_sample
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


@pytest.fixture(scope="module")
def small_flow():
    """The XS problem with a deliberately tiny network: the integrator is under test."""
    with open(Path(lisa.__file__).parent / "configs" / "XS.yaml") as f:
        problem = LisaGB(**yaml.safe_load(f)["problem"])
    sample = train_sample(problem, jr.key(0))
    flow = LisaFlow(
        x_shape=sample.xt.shape,
        y_shape=sample.y.shape,
        hidden_dim=32,
        num_heads=2,
        num_blocks=1,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        key=jr.key(1),
    )
    # a fresh MMDiT is zero-initialised into the identity transport; perturb it so the
    # velocity field actually depends on x, t and y
    params, static = jax.tree_util.tree_flatten(flow)
    keys = jr.split(jr.key(2), len(params))
    params = [
        p + 0.05 * jr.normal(k, p.shape, p.dtype)
        if isinstance(p, jax.Array) and jnp.issubdtype(p.dtype, jnp.floating)
        else p
        for p, k in zip(params, keys)
    ]
    flow = jax.tree_util.tree_unflatten(static, params)
    u0 = jax.vmap(problem.sample_flow, in_axes=(0, None))(jr.split(jr.key(3), 8), sample.f)
    return problem, flow, u0, sample.y, sample.f


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
