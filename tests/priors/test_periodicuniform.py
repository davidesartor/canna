"""PeriodicUniform: uniform over [0,period)^D, Periodic chart, Toroidal geometry."""

import jax
import jax.numpy as jnp

from canna.geometries import Toroidal
from canna.charts import Periodic
from canna.priors import PeriodicUniform


def test_geometry_is_toroidal():
    assert isinstance(PeriodicUniform().geometry, Toroidal)


def test_chart_is_periodic():
    assert isinstance(PeriodicUniform().chart, Periodic)


def test_default_period_shape_and_value():
    prior = PeriodicUniform()
    assert prior.period.shape == (1,)
    assert jnp.allclose(prior.period, 2 * jnp.pi)


def test_chart_period_matches_prior_period():
    prior = PeriodicUniform(period=jnp.array([2 * jnp.pi, 4.0]))
    assert jnp.allclose(prior.chart.period, prior.period)


def test_call_shape_matches_period_length():
    prior = PeriodicUniform(period=jnp.array([2 * jnp.pi, 4.0, 1.0]))
    sample = prior(jax.random.key(0))
    assert sample.shape == (3,)


def test_call_within_period_bounds_per_axis():
    prior = PeriodicUniform(period=jnp.array([2 * jnp.pi, 4.0]))
    keys = jax.random.split(jax.random.key(1), 500)
    samples = jax.vmap(prior)(keys)
    assert jnp.all(samples[:, 0] >= 0.0) and jnp.all(samples[:, 0] < 2 * jnp.pi)
    assert jnp.all(samples[:, 1] >= 0.0) and jnp.all(samples[:, 1] < 4.0)


def test_call_is_deterministic_given_key():
    prior = PeriodicUniform()
    key = jax.random.key(42)
    assert jnp.allclose(prior(key), prior(key))


def test_call_varies_across_keys():
    prior = PeriodicUniform(period=jnp.array([2 * jnp.pi, 4.0]))
    s0 = prior(jax.random.key(0))
    s1 = prior(jax.random.key(1))
    assert not jnp.allclose(s0, s1)


def test_embedded_draws_on_manifold():
    prior = PeriodicUniform(period=jnp.array([2 * jnp.pi, 4.0]))
    keys = jax.random.split(jax.random.key(2), 50)
    samples = jax.vmap(prior)(keys)
    embedded = jax.vmap(prior.chart.forward)(samples)
    assert jnp.allclose(jnp.sum(embedded**2, axis=-1), 2.0, atol=1e-3)


def test_default_dim_one_sample_shape():
    prior = PeriodicUniform()
    sample = prior(jax.random.key(3))
    assert sample.shape == (1,)


def test_uniform_mean_near_period_half_over_many_draws():
    # weak distributional sanity: uniform on [0,period) has mean period/2
    prior = PeriodicUniform(period=jnp.array([2 * jnp.pi]))
    keys = jax.random.split(jax.random.key(4), 5000)
    samples = jax.vmap(prior)(keys)
    assert jnp.abs(jnp.mean(samples) - jnp.pi) < 0.15


def test_call_matches_direct_jr_uniform_reference():
    prior = PeriodicUniform(period=jnp.array([2 * jnp.pi, 3.0]))
    key = jax.random.key(7)
    expected = jax.random.uniform(key, prior.period.shape, maxval=prior.period)
    assert jnp.allclose(prior(key), expected, atol=1e-6)
