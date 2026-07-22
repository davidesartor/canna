"""Shape-signature contracts for the periodic/spherical/set concretes."""

import jax
import jax.numpy as jnp
import pytest

from canna import geometries, charts, priors


@pytest.mark.parametrize("D", [1, 3])
def test_periodic_chart_roundtrip_shapes(D):
    chart = charts.Periodic(jnp.full((D,), 2 * jnp.pi))
    p = jax.random.uniform(jax.random.key(0), (D,), maxval=2 * jnp.pi)
    x = chart.forward(p)
    assert x.shape == (2 * D,)
    assert chart.backward(x).shape == (D,)


@pytest.mark.parametrize("D", [2, 3])
def test_spherical_chart_roundtrip_shapes(D):
    chart = charts.Spherical(D, jnp.array(1.0))
    p = jax.random.uniform(jax.random.key(0), (D,))
    x = chart.forward(p)
    assert x.shape == (D + 1,)
    assert chart.backward(x).shape == (D,)


@pytest.mark.parametrize("batch_shape", [(), (4,), (2, 3)])
def test_toroidal_maps_preserve_shape(batch_shape):
    D = 2
    geo = geometries.Toroidal(2 * D)
    k0, k1 = jax.random.split(jax.random.key(0))
    x0 = jax.random.normal(k0, batch_shape + (2 * D,))
    x1 = jax.random.normal(k1, batch_shape + (2 * D,))
    assert geo.log_map(x0, x1).shape == batch_shape + (2 * D,)
    assert geo.exp_map(x0, geo.log_map(x0, x1)).shape == batch_shape + (2 * D,)
    assert geo.geodesic(jnp.array(0.3), x0, x1).shape == batch_shape + (2 * D,)


@pytest.mark.parametrize("batch_shape", [(), (4,), (2, 3)])
def test_spherical_geometry_maps_preserve_shape(batch_shape):
    D = 3
    geo = geometries.Spherical(D)
    k0, k1 = jax.random.split(jax.random.key(0))
    x0 = jax.random.normal(k0, batch_shape + (D,))
    x1 = jax.random.normal(k1, batch_shape + (D,))
    assert geo.log_map(x0, x1).shape == batch_shape + (D,)
    assert geo.exp_map(x0, geo.log_map(x0, x1)).shape == batch_shape + (D,)


@pytest.mark.parametrize("batch_shape", [(), (5,)])
def test_set_geometry_maps_and_assign_shapes(batch_shape):
    S, X = 4, 3
    geo = geometries.Set(geometries.Euclidean())
    k0, k1 = jax.random.split(jax.random.key(0))
    x0 = jax.random.normal(k0, batch_shape + (S, X))
    x1 = jax.random.normal(k1, batch_shape + (S, X))
    assert geo.log_map(x0, x1).shape == batch_shape + (S, X)
    assert geo.exp_map(x0, geo.log_map(x0, x1)).shape == batch_shape + (S, X)
    assert geo.assign(x0, x1).shape == batch_shape + (S, X)
    assert geo.assign_by_rank(x0, x1).shape == batch_shape + (S,)
    assert geo.assign_by_brute_force(x0, x1).shape == batch_shape + (S,)


def test_product_geometry_preserves_total_dim():
    geo = geometries.Product(geometries.Euclidean(), geometries.Spherical(3))
    D = 2 + 3
    k0, k1 = jax.random.split(jax.random.key(0))
    x0 = jax.random.normal(k0, (D,))
    x1 = jax.random.normal(k1, (D,))
    assert geo.log_map(x0, x1).shape == (D,)
    assert geo.exp_map(x0, geo.log_map(x0, x1)).shape == (D,)


@pytest.mark.parametrize("D", [1, 3])
def test_periodicuniform_call_shape(D):
    prior = priors.PeriodicUniform(jnp.full((D,), 2 * jnp.pi))
    assert prior(jax.random.key(0)).shape == (D,)
    assert isinstance(prior.geometry, geometries.Toroidal)


@pytest.mark.parametrize("dim", [1, 3])
def test_isotropic_call_shape(dim):
    prior = priors.Isotropic(dim=dim)
    assert prior(jax.random.key(0)).shape == (dim,)
    assert isinstance(prior.geometry, geometries.Spherical)


def test_product_prior_call_shape():
    prior = priors.Product(priors.Normal(mean=jnp.zeros(2)), priors.Isotropic(dim=1))
    assert prior(jax.random.key(0)).shape == (2 + 1,)


@pytest.mark.parametrize("S", [2, 4])
def test_set_prior_call_shape(S):
    prior = priors.Set(priors.Normal(mean=jnp.zeros(3)), size=S)
    assert prior(jax.random.key(0)).shape == (S, 3)
    assert isinstance(prior.geometry, geometries.Set)
