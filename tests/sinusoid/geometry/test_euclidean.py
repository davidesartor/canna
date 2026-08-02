"""Euclidean: flat geodesics, log/exp inverses, batch shape."""

import jax
import jax.numpy as jnp
import pytest

from canna.sinusoid.geometries import Euclidean


@pytest.fixture
def geo():
    return Euclidean()


@pytest.mark.parametrize("D", [1, 2, 5])
def test_geodesic_endpoint_t0_is_x0(geo, D):
    x0 = jnp.arange(D, dtype=jnp.float32)
    x1 = jnp.arange(D, dtype=jnp.float32) + 3.7
    out = geo.geodesic(jnp.array(0.0), x0, x1)
    assert jnp.allclose(out, x0)


@pytest.mark.parametrize("D", [1, 2, 5])
def test_geodesic_endpoint_t1_is_x1(geo, D):
    x0 = jnp.arange(D, dtype=jnp.float32)
    x1 = jnp.arange(D, dtype=jnp.float32) + 3.7
    out = geo.geodesic(jnp.array(1.0), x0, x1)
    assert jnp.allclose(out, x1)


def test_geodesic_linear_formula_matches_lerp(geo):
    x0 = jnp.array([0.0, -2.0, 5.0])
    x1 = jnp.array([1.0, 4.0, -3.0])
    for t in [0.1, 0.25, 0.5, 0.73, 0.9]:
        expected = x0 + t * (x1 - x0)
        out = geo.geodesic(jnp.array(t), x0, x1)
        assert jnp.allclose(out, expected, atol=1e-5)


def test_geodesic_reversal_symmetry(geo):
    x0 = jnp.array([1.0, 2.0, 3.0])
    x1 = jnp.array([-4.0, 0.5, 9.0])
    for t in [0.0, 0.3, 0.5, 1.0]:
        forward = geo.geodesic(jnp.array(t), x0, x1)
        backward = geo.geodesic(jnp.array(1.0 - t), x1, x0)
        assert jnp.allclose(forward, backward, atol=1e-5)


def test_geodesic_degenerate_coincident_endpoints(geo):
    x0 = jnp.array([2.0, -1.0, 0.0])
    for t in [0.0, 0.3, 0.5, 1.0]:
        out = geo.geodesic(jnp.array(t), x0, x0)
        assert jnp.allclose(out, x0)


def test_log_map_zero_at_coincident_points(geo):
    x0 = jnp.array([1.0, -3.0, 2.5])
    out = geo.log_map(x0, x0)
    assert jnp.allclose(out, jnp.zeros_like(x0))


def test_exp_map_zero_tangent_is_identity(geo):
    x0 = jnp.array([1.0, -3.0, 2.5])
    zero = jnp.zeros_like(x0)
    out = geo.exp_map(x0, zero)
    assert jnp.allclose(out, x0)


def test_log_exp_map_are_inverses(geo):
    x0 = jnp.array([0.0, 0.0, 0.0])
    x1 = jnp.array([1.0, -2.0, 3.5])
    dx = geo.log_map(x0, x1)
    out = geo.exp_map(x0, dx)
    assert jnp.allclose(out, x1, atol=1e-5)


def test_log_map_antisymmetric(geo):
    x0 = jnp.array([1.0, 2.0, 3.0])
    x1 = jnp.array([-1.0, 5.0, 0.0])
    assert jnp.allclose(geo.log_map(x0, x1), -geo.log_map(x1, x0), atol=1e-5)


def test_geodesic_matches_exp_of_scaled_log(geo):
    x0 = jnp.array([1.0, -1.0, 2.0])
    x1 = jnp.array([4.0, 0.0, -2.0])
    for t in [0.0, 0.2, 0.6, 1.0]:
        via_geodesic = geo.geodesic(jnp.array(t), x0, x1)
        via_log_exp = geo.exp_map(x0, jnp.array(t) * geo.log_map(x0, x1))
        assert jnp.allclose(via_geodesic, via_log_exp, atol=1e-5)


@pytest.mark.parametrize("batch_shape", [(), (4,), (2, 3)])
def test_geodesic_batch_shape_preserved(geo, batch_shape):
    D = 5
    key = jax.random.key(0)
    k0, k1 = jax.random.split(key)
    x0 = jax.random.normal(k0, batch_shape + (D,))
    x1 = jax.random.normal(k1, batch_shape + (D,))
    out = geo.geodesic(jnp.array(0.37), x0, x1)
    assert out.shape == batch_shape + (D,)


@pytest.mark.parametrize("batch_shape", [(), (4,), (2, 3)])
def test_log_map_batch_shape_preserved(geo, batch_shape):
    D = 5
    key = jax.random.key(1)
    k0, k1 = jax.random.split(key)
    x0 = jax.random.normal(k0, batch_shape + (D,))
    x1 = jax.random.normal(k1, batch_shape + (D,))
    out = geo.log_map(x0, x1)
    assert out.shape == batch_shape + (D,)
