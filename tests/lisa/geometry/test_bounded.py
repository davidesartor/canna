"""Bounded: box [-1,1]^D, geodesics are straight lines clipped to the box."""

import jax
import jax.numpy as jnp
import pytest

from canna.lisa.geometries import Bounded, Euclidean


@pytest.fixture
def geo():
    return Bounded()


def test_log_map_matches_euclidean_interior(geo):
    x0 = jnp.array([0.1, -0.2])
    x1 = jnp.array([0.5, 0.3])
    assert jnp.allclose(geo.log_map(x0, x1), x1 - x0)


def test_log_map_zero_for_coincident_points(geo):
    x0 = jnp.array([0.3, -0.7])
    assert jnp.allclose(geo.log_map(x0, x0), jnp.zeros_like(x0))


def test_exp_map_interior_matches_euclidean(geo):
    x0 = jnp.array([0.0, 0.2])
    dx = jnp.array([0.3, -0.1])
    assert jnp.allclose(geo.exp_map(x0, dx), Euclidean().exp_map(x0, dx))


def test_exp_map_clips_above(geo):
    x0 = jnp.array([0.9])
    dx = jnp.array([0.5])
    out = geo.exp_map(x0, dx)
    assert jnp.allclose(out, jnp.array([1.0]))


def test_exp_map_clips_below(geo):
    x0 = jnp.array([-0.9])
    dx = jnp.array([-0.5])
    out = geo.exp_map(x0, dx)
    assert jnp.allclose(out, jnp.array([-1.0]))


def test_exp_map_boundary_zero_step_is_identity(geo):
    x0 = jnp.array([1.0, -1.0])
    dx = jnp.zeros_like(x0)
    assert jnp.allclose(geo.exp_map(x0, dx), x0)


def test_exp_map_far_overflow_saturates_at_boundary(geo):
    x0 = jnp.array([0.0])
    dx = jnp.array([100.0])
    out = geo.exp_map(x0, dx)
    assert jnp.allclose(out, jnp.array([1.0]))


def test_geodesic_endpoints(geo):
    x0 = jnp.array([-0.5, 0.2])
    x1 = jnp.array([0.4, -0.6])
    assert jnp.allclose(geo.geodesic(jnp.asarray(0.0), x0, x1), x0)
    assert jnp.allclose(geo.geodesic(jnp.asarray(1.0), x0, x1), x1)


def test_geodesic_identity_when_x0_equals_x1(geo):
    x = jnp.array([0.3, -0.2])
    for t in jnp.array([0.0, 0.25, 0.5, 1.0]):
        assert jnp.allclose(geo.geodesic(t, x, x), x)


def test_geodesic_stays_in_box_for_interior_endpoints(geo):
    x0 = jnp.array([-1.0, 1.0])
    x1 = jnp.array([1.0, -1.0])
    for t in jnp.linspace(0.0, 1.0, 11):
        p = geo.geodesic(t, x0, x1)
        assert jnp.all(p >= -1.0) and jnp.all(p <= 1.0)


def test_geodesic_midpoint_between_opposite_corners_is_center(geo):
    x0 = jnp.array([-1.0])
    x1 = jnp.array([1.0])
    out = geo.geodesic(jnp.asarray(0.5), x0, x1)
    assert jnp.allclose(out, jnp.array([0.0]))


@pytest.mark.parametrize("batch_shape", [(), (2, 3)])
def test_exp_map_leading_batch_dims_preserved(geo, batch_shape):
    D = 4
    x0 = jnp.zeros(batch_shape + (D,))
    dx = jnp.ones(batch_shape + (D,)) * 0.1
    out = geo.exp_map(x0, dx)
    assert out.shape == batch_shape + (D,)


def test_exp_map_clip_formula_matches_source(geo):
    x0 = jnp.array([0.2, -0.6, 0.95])
    dx = jnp.array([2.0, -3.0, 0.02])
    expected = jnp.clip(x0 + dx, -1.0, 1.0)
    assert jnp.allclose(geo.exp_map(x0, dx), expected)


def test_exp_map_exactly_at_boundary_no_overshoot(geo):
    x0 = jnp.array([0.0])
    dx = jnp.array([1.0])
    out = geo.exp_map(x0, dx)
    assert jnp.allclose(out, jnp.array([1.0]))


@pytest.mark.parametrize("batch_shape", [(4,)])
def test_exp_map_random_batch_clip_matches_reference(geo, batch_shape):
    D = 3
    key = jax.random.key(0)
    k0, k1 = jax.random.split(key)
    x0 = jax.random.uniform(k0, batch_shape + (D,), minval=-1.0, maxval=1.0)
    dx = jax.random.uniform(k1, batch_shape + (D,), minval=-3.0, maxval=3.0)
    out = geo.exp_map(x0, dx)
    expected = jnp.clip(x0 + dx, -1.0, 1.0)
    assert jnp.allclose(out, expected)
