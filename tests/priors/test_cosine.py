"""Cosine: density ~ cos over [-pi/2,pi/2] (declination), Bounded geometry, Affine chart."""

import jax
import jax.numpy as jnp

from canna.geometries import Bounded
from canna.charts import Affine
from canna.priors import Cosine


def test_geometry_is_bounded():
    assert isinstance(Cosine().geometry, Bounded)


def test_chart_is_affine():
    assert isinstance(Cosine().chart, Affine)


def test_chart_maps_neg_half_pi_to_neg_one():
    prior = Cosine()
    out = prior.chart.forward(jnp.array([-jnp.pi / 2]))
    assert jnp.allclose(out, -1.0)


def test_chart_maps_pos_half_pi_to_pos_one():
    prior = Cosine()
    out = prior.chart.forward(jnp.array([jnp.pi / 2]))
    assert jnp.allclose(out, 1.0)


def test_chart_maps_zero_to_zero():
    prior = Cosine()
    out = prior.chart.forward(jnp.array([0.0]))
    assert jnp.allclose(out, 0.0)


def test_chart_roundtrip():
    prior = Cosine()
    p = jnp.array([0.3])
    assert jnp.allclose(prior.chart.backward(prior.chart.forward(p)), p)


def test_call_within_declination_bounds():
    prior = Cosine()
    keys = jax.random.split(jax.random.key(0), 500)
    samples = jax.vmap(prior)(keys)
    assert jnp.all(samples >= -jnp.pi / 2) and jnp.all(samples <= jnp.pi / 2)


def test_call_is_not_uniform_on_domain():
    # density ~ cos concentrates mass near 0, not near the +-pi/2 edges
    prior = Cosine()
    keys = jax.random.split(jax.random.key(0), 2000)
    samples = jax.vmap(prior)(keys)
    frac_near_edges = jnp.mean(jnp.abs(jnp.abs(samples) - jnp.pi / 2) < 0.1)
    frac_near_center = jnp.mean(jnp.abs(samples) < 0.1)
    assert frac_near_center > frac_near_edges


def test_default_dim_is_1():
    prior = Cosine()
    sample = prior(jax.random.key(0))
    assert sample.shape == (1,)


def test_dim_generalizes_call_shape():
    prior = Cosine(dim=3)
    sample = prior(jax.random.key(0))
    assert sample.shape == (3,)


def test_dim_generalizes_call_within_bounds_all_axes():
    prior = Cosine(dim=4)
    keys = jax.random.split(jax.random.key(1), 300)
    samples = jax.vmap(prior)(keys)
    assert samples.shape == (300, 4)
    assert jnp.all(samples >= -jnp.pi / 2) and jnp.all(samples <= jnp.pi / 2)


def test_dim_generalizes_chart_shift_and_scale_length():
    prior = Cosine(dim=5)
    assert prior.chart.shift.shape == (5,)
    assert prior.chart.scale.shape == (5,)


def test_dim_generalizes_chart_endpoint_maps_per_axis():
    prior = Cosine(dim=3)
    low_corner = jnp.full((3,), -jnp.pi / 2)
    high_corner = jnp.full((3,), jnp.pi / 2)
    assert jnp.allclose(prior.chart.forward(low_corner), -1.0)
    assert jnp.allclose(prior.chart.forward(high_corner), 1.0)


def test_dim_generalizes_chart_roundtrip_vector():
    prior = Cosine(dim=3)
    p = jnp.array([0.2, -1.0, 0.7])
    assert jnp.allclose(prior.chart.backward(prior.chart.forward(p)), p)


def test_dim_axes_are_independent_draws():
    # each axis is an independent arcsin(uniform) draw, not a repeated scalar
    prior = Cosine(dim=8)
    sample = prior(jax.random.key(2))
    assert jnp.unique(sample).shape[0] > 1
