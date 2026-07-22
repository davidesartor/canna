"""Uniform: draws on [low,high], Bounded geometry, Affine chart keeping it uniform."""

import jax
import jax.numpy as jnp

from canna.geometries import Bounded
from canna.charts import Affine
from canna.priors import Uniform


def test_geometry_is_bounded():
    assert isinstance(Uniform(low=-2.0, high=3.0).geometry, Bounded)


def test_chart_is_affine():
    assert isinstance(Uniform(low=-2.0, high=3.0).chart, Affine)


def test_chart_maps_low_to_neg_one():
    prior = Uniform(low=-2.0, high=3.0)
    out = prior.chart.forward(prior.low)
    assert jnp.allclose(out, -1.0)


def test_chart_maps_high_to_pos_one():
    prior = Uniform(low=-2.0, high=3.0)
    out = prior.chart.forward(prior.high)
    assert jnp.allclose(out, 1.0)


def test_chart_maps_midpoint_to_zero():
    prior = Uniform(low=-2.0, high=3.0)
    mid = (prior.low + prior.high) / 2
    out = prior.chart.forward(mid)
    assert jnp.allclose(out, 0.0)


def test_chart_roundtrip():
    prior = Uniform(low=jnp.array([-2.0, 1.0]), high=jnp.array([3.0, 5.0]))
    p = jnp.array([0.0, 2.0])
    assert jnp.allclose(prior.chart.backward(prior.chart.forward(p)), p)


def test_chart_scale_is_elementwise_not_matrix():
    prior = Uniform(low=jnp.array([-2.0, 1.0]), high=jnp.array([3.0, 5.0]))
    assert prior.chart.scale.ndim < 2


def test_call_within_bounds():
    prior = Uniform(low=-2.0, high=3.0)
    sample = prior(jax.random.key(0))
    assert jnp.all(sample >= -2.0) and jnp.all(sample <= 3.0)


def test_call_shape_matches_dimension():
    prior = Uniform(low=jnp.array([-2.0, 1.0, 0.0]), high=jnp.array([3.0, 5.0, 1.0]))
    sample = prior(jax.random.key(0))
    assert sample.shape == (3,)


def test_call_batched_over_keys_stays_within_bounds():
    prior = Uniform(low=-2.0, high=3.0)
    keys = jax.random.split(jax.random.key(0), 200)
    samples = jax.vmap(prior)(keys)
    assert samples.shape == (200, 1)
    assert jnp.all(samples >= -2.0) and jnp.all(samples <= 3.0)


def test_call_is_stochastic():
    prior = Uniform(low=-2.0, high=3.0)
    a = prior(jax.random.key(1))
    b = prior(jax.random.key(2))
    assert not jnp.allclose(a, b)


def test_default_low_high_are_0_and_1():
    prior = Uniform()
    assert jnp.allclose(prior.low, 0.0) and jnp.allclose(prior.high, 1.0)


def test_chart_scale_and_shift_match_exact_formula():
    low = jnp.array([-2.0, 1.0, 4.0])
    high = jnp.array([3.0, 5.0, 10.0])
    prior = Uniform(low=low, high=high)
    expected_scale = 2 / (high - low)
    expected_shift = -expected_scale * (low + high) / 2
    assert jnp.allclose(prior.chart.scale, expected_scale)
    assert jnp.allclose(prior.chart.shift, expected_shift)


def test_call_shape_follows_low_even_when_high_is_scalar():
    # jr.uniform is called with self.low.shape as the sample shape, so a
    # vector low broadcast against a scalar high still yields low's shape.
    prior = Uniform(low=jnp.array([0.0, 1.0, 2.0]), high=5.0)
    sample = prior(jax.random.key(0))
    assert sample.shape == (3,)


def test_call_batch_mean_approaches_midpoint():
    prior = Uniform(low=-2.0, high=3.0)
    keys = jax.random.split(jax.random.key(42), 4000)
    samples = jax.vmap(prior)(keys)
    assert jnp.allclose(jnp.mean(samples), 0.5, atol=0.1)


def test_call_scalar_low_high_broadcast_but_shape_is_1d_not_scalar():
    prior = Uniform(low=0.0, high=1.0)
    sample = prior(jax.random.key(0))
    assert sample.shape == (1,)
