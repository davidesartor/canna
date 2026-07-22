"""LogUniform: draws log-uniform on [low,high], Bounded geometry, LogAffine chart."""

import jax
import jax.numpy as jnp
import pytest

from canna.geometries import Bounded
from canna.charts import LogAffine
from canna.priors import LogUniform


def test_geometry_is_bounded():
    assert isinstance(LogUniform(low=0.5, high=4.0).geometry, Bounded)


def test_chart_is_logaffine():
    assert isinstance(LogUniform(low=0.5, high=4.0).chart, LogAffine)


def test_chart_maps_low_to_neg_one():
    prior = LogUniform(low=0.5, high=4.0)
    out = prior.chart.forward(prior.low)
    assert jnp.allclose(out, -1.0)


def test_chart_maps_high_to_pos_one():
    prior = LogUniform(low=0.5, high=4.0)
    out = prior.chart.forward(prior.high)
    assert jnp.allclose(out, 1.0)


def test_chart_roundtrip():
    prior = LogUniform(low=jnp.array([0.5, 2.0]), high=jnp.array([4.0, 10.0]))
    p = jnp.array([1.0, 5.0])
    assert jnp.allclose(prior.chart.backward(prior.chart.forward(p)), p)


def test_call_within_bounds_and_positive():
    prior = LogUniform(low=0.5, high=4.0)
    sample = prior(jax.random.key(0))
    assert jnp.all(sample > 0.0)
    assert jnp.all(sample >= 0.5) and jnp.all(sample <= 4.0)


def test_call_shape_matches_dimension():
    prior = LogUniform(low=jnp.array([0.5, 2.0, 1.0]), high=jnp.array([4.0, 10.0, 2.0]))
    sample = prior(jax.random.key(0))
    assert sample.shape == (3,)


def test_requires_low_and_high_no_defaults():
    with pytest.raises(TypeError):
        LogUniform()


def test_chart_scale_and_shift_match_exact_log_space_formula():
    low = jnp.array([0.5, 2.0, 1.0])
    high = jnp.array([4.0, 10.0, 2.0])
    prior = LogUniform(low=low, high=high)
    log_low, log_high = jnp.log(low), jnp.log(high)
    expected_scale = 2 / (log_high - log_low)
    expected_shift = -expected_scale * (log_low + log_high) / 2
    assert jnp.allclose(prior.chart.scale, expected_scale)
    assert jnp.allclose(prior.chart.shift, expected_shift)


def test_call_is_geometric_mean_biased_not_arithmetic_uniform():
    # log-uniform sample mean should track the geometric-ish log-space midpoint,
    # not the arithmetic midpoint of low/high.
    prior = LogUniform(low=0.5, high=4.0)
    keys = jax.random.split(jax.random.key(7), 4000)
    samples = jax.vmap(prior)(keys)
    arithmetic_mid = (0.5 + 4.0) / 2
    assert jnp.mean(samples) < arithmetic_mid


def test_call_undefined_for_nonpositive_low_produces_nan_or_inf():
    # log(low) for low<=0 is undefined; the implementation does not validate
    # low>0, so this documents the (unchecked) failure mode.
    prior = LogUniform(low=jnp.array([-1.0]), high=jnp.array([2.0]))
    sample = prior(jax.random.key(0))
    assert jnp.isnan(sample).any() or jnp.isinf(sample).any()
