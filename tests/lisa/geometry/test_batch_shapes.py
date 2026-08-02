"""Shape-signature contracts for the periodic/spherical/set geometries."""

import jax
import jax.numpy as jnp
import pytest

from canna.lisa import geometries


@pytest.mark.parametrize("batch_shape", [(), (4,), (2, 3)])
def test_circle_maps_preserve_shape(batch_shape):
    D = 2
    geo = geometries.Spherical(D)
    k0, k1 = jax.random.split(jax.random.key(0))
    x0 = jax.random.normal(k0, batch_shape + (D,))
    x1 = jax.random.normal(k1, batch_shape + (D,))
    assert geo.log_map(x0, x1).shape == batch_shape + (D,)
    assert geo.exp_map(x0, geo.log_map(x0, x1)).shape == batch_shape + (D,)
    assert geo.geodesic(jnp.array(0.3), x0, x1).shape == batch_shape + (D,)


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
