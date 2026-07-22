"""Sine: density ~ sin over [0,pi] (inclination), Reflected geometry, Affine chart."""

import jax
import jax.numpy as jnp

from canna.geometries import Reflected
from canna.charts import Affine
from canna.priors import Sine


def test_geometry_is_reflected():
    assert isinstance(Sine().geometry, Reflected)


def test_chart_is_affine():
    assert isinstance(Sine().chart, Affine)


def test_chart_maps_zero_to_neg_one():
    prior = Sine()
    out = prior.chart.forward(jnp.array([0.0]))
    assert jnp.allclose(out, -1.0)


def test_chart_maps_pi_to_pos_one():
    prior = Sine()
    out = prior.chart.forward(jnp.array([jnp.pi]))
    assert jnp.allclose(out, 1.0)


def test_chart_roundtrip():
    prior = Sine()
    p = jnp.array([1.2])
    assert jnp.allclose(prior.chart.backward(prior.chart.forward(p)), p)


def test_call_within_inclination_bounds():
    prior = Sine()
    keys = jax.random.split(jax.random.key(0), 500)
    samples = jax.vmap(prior)(keys)
    assert jnp.all(samples >= 0.0) and jnp.all(samples <= jnp.pi)


def test_call_is_not_uniform_on_domain():
    # density ~ sin concentrates mass near pi/2, away from 0 and pi edges
    prior = Sine()
    keys = jax.random.split(jax.random.key(0), 2000)
    samples = jax.vmap(prior)(keys)
    frac_near_edges = jnp.mean((samples < 0.1) | (samples > jnp.pi - 0.1))
    frac_near_center = jnp.mean(jnp.abs(samples - jnp.pi / 2) < 0.1)
    assert frac_near_center > frac_near_edges


def test_default_dim_is_1():
    prior = Sine()
    sample = prior(jax.random.key(0))
    assert sample.shape == (1,)


def test_dim_generalizes_call_shape():
    prior = Sine(dim=3)
    sample = prior(jax.random.key(0))
    assert sample.shape == (3,)


def test_dim_generalizes_call_within_bounds_all_axes():
    prior = Sine(dim=4)
    keys = jax.random.split(jax.random.key(1), 300)
    samples = jax.vmap(prior)(keys)
    assert samples.shape == (300, 4)
    assert jnp.all(samples >= 0.0) and jnp.all(samples <= jnp.pi)


def test_dim_generalizes_chart_shift_and_scale_length():
    prior = Sine(dim=5)
    assert prior.chart.shift.shape == (5,)
    assert prior.chart.scale.shape == (5,)


def test_dim_generalizes_chart_endpoint_maps_per_axis():
    prior = Sine(dim=3)
    low_corner = jnp.zeros(3)
    high_corner = jnp.full((3,), jnp.pi)
    assert jnp.allclose(prior.chart.forward(low_corner), -1.0)
    assert jnp.allclose(prior.chart.forward(high_corner), 1.0)


def test_dim_generalizes_chart_roundtrip_vector():
    prior = Sine(dim=3)
    p = jnp.array([0.2, 1.0, 2.7])
    assert jnp.allclose(prior.chart.backward(prior.chart.forward(p)), p)


def test_dim_axes_are_independent_draws():
    prior = Sine(dim=8)
    sample = prior(jax.random.key(2))
    assert jnp.unique(sample).shape[0] > 1
