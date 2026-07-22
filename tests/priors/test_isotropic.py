"""Isotropic: uniform over sphere surface in R^{dim+1}, Spherical chart+geometry."""

import jax
import jax.numpy as jnp

from canna.geometries import Spherical as SphericalGeometry
from canna.charts import Spherical as SphericalChart
from canna.priors import Isotropic


def test_geometry_is_spherical():
    assert isinstance(Isotropic().geometry, SphericalGeometry)


def test_chart_is_spherical():
    assert isinstance(Isotropic().chart, SphericalChart)


def test_chart_radius_matches_prior_radius():
    prior = Isotropic(dim=2, radius=3.0)
    assert jnp.allclose(prior.chart.radius, 3.0)


def test_default_dim_is_1_default_radius_is_1():
    prior = Isotropic()
    assert prior.dim == 1
    assert jnp.allclose(prior.radius, 1.0)


def test_call_shape_matches_dim():
    prior = Isotropic(dim=3)
    sample = prior(jax.random.key(0))
    assert sample.shape == (3,)


def test_call_is_deterministic_given_key():
    prior = Isotropic(dim=2)
    key = jax.random.key(42)
    assert jnp.allclose(prior(key), prior(key))


def test_call_varies_across_keys():
    prior = Isotropic(dim=2)
    s0 = prior(jax.random.key(0))
    s1 = prior(jax.random.key(1))
    assert not jnp.allclose(s0, s1)


def test_embedded_draws_lie_on_sphere_of_declared_radius():
    prior = Isotropic(dim=2, radius=2.0)
    keys = jax.random.split(jax.random.key(1), 50)
    samples = jax.vmap(prior)(keys)
    embedded = jax.vmap(prior.chart.forward)(samples)
    assert embedded.shape == (50, 3)
    norms = jnp.linalg.norm(embedded, axis=-1)
    assert jnp.allclose(norms, 2.0, atol=1e-3)


def test_embedded_draws_lie_on_unit_sphere_default_radius():
    prior = Isotropic(dim=1)
    keys = jax.random.split(jax.random.key(2), 50)
    samples = jax.vmap(prior)(keys)
    embedded = jax.vmap(prior.chart.forward)(samples)
    norms = jnp.linalg.norm(embedded, axis=-1)
    assert jnp.allclose(norms, 1.0, atol=1e-3)


def test_isotropic_mean_embedded_position_near_origin():
    # uniform-on-sphere-surface samples should have ~zero mean direction
    prior = Isotropic(dim=2)
    keys = jax.random.split(jax.random.key(3), 4000)
    samples = jax.vmap(prior)(keys)
    embedded = jax.vmap(prior.chart.forward)(samples)
    mean_pos = jnp.mean(embedded, axis=0)
    assert jnp.all(jnp.abs(mean_pos) < 0.1)


def test_isotropic_dim1_reduces_to_uniform_angle_on_circle():
    # dim=1 -> one angle, embeds to unit circle; angle should range over full
    # [0, 2pi) or (-pi, pi], not be confined to a sub-arc.
    prior = Isotropic(dim=1)
    keys = jax.random.split(jax.random.key(4), 2000)
    samples = jax.vmap(prior)(keys)
    spread = jnp.max(samples) - jnp.min(samples)
    assert spread > 5.0


def test_call_matches_direct_normalized_gaussian_reference():
    prior = Isotropic(dim=2, radius=1.5)
    key = jax.random.key(11)
    v = jax.random.normal(key, (3,))
    expected_embedded = 1.5 * v / jnp.linalg.norm(v)
    expected = prior.chart.backward(expected_embedded)
    assert jnp.allclose(prior(key), expected, atol=1e-5)


def test_call_embedded_matches_normalized_gaussian_exactly():
    # chart.forward(prior(key)) should reproduce the same
    # embedded point __call__ derives from jr.normal, not merely
    # "some point on the sphere".
    prior = Isotropic(dim=2, radius=2.0)
    key = jax.random.key(12)
    v = jax.random.normal(key, (3,))
    expected_embedded = 2.0 * v / jnp.linalg.norm(v)
    sample = prior(key)
    assert jnp.allclose(prior.chart.forward(sample), expected_embedded, atol=1e-4)
