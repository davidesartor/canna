"""Reflected: box [-1,1]^D with reflecting (triangle-wave) boundaries."""

import jax
import jax.numpy as jnp
import pytest

from canna.geometries import Reflected, Euclidean, Bounded


def reflect_into_box(y):
    """Slow triangle-wave reference: fold y into [-1, 1] via reflecting boundaries."""
    wrapped = jnp.mod(y + 1.0, 4.0) - 1.0
    return jnp.where(wrapped > 1.0, 2.0 - wrapped, wrapped)


@pytest.fixture
def geo():
    return Reflected()


def test_log_map_matches_euclidean_interior(geo):
    x0 = jnp.array([0.1, -0.2])
    x1 = jnp.array([0.5, 0.3])
    assert jnp.allclose(geo.log_map(x0, x1), x1 - x0)


def test_exp_map_interior_matches_euclidean(geo):
    x0 = jnp.array([0.0, 0.2])
    dx = jnp.array([0.3, -0.1])
    assert jnp.allclose(geo.exp_map(x0, dx), Euclidean().exp_map(x0, dx))


def test_exp_map_single_bounce_above(geo):
    x0 = jnp.array([0.9])
    dx = jnp.array([0.5])
    out = geo.exp_map(x0, dx)
    assert jnp.allclose(out, jnp.array([0.6]))


def test_exp_map_single_bounce_below(geo):
    x0 = jnp.array([-0.9])
    dx = jnp.array([-0.5])
    out = geo.exp_map(x0, dx)
    assert jnp.allclose(out, jnp.array([-0.6]))


def test_exp_map_multiple_bounces_matches_triangle_wave_reference(geo):
    x0 = jnp.array([0.2])
    dx = jnp.array([5.3])
    expected = reflect_into_box(x0 + dx)
    out = geo.exp_map(x0, dx)
    assert jnp.allclose(out, expected)


def test_exp_map_result_always_in_box(geo):
    x0 = jnp.array([0.0, -0.4, 1.0])
    dx = jnp.array([10.3, -7.7, 3.0])
    out = geo.exp_map(x0, dx)
    assert jnp.all(out >= -1.0) and jnp.all(out <= 1.0)


def test_exp_map_boundary_zero_step_is_identity(geo):
    x0 = jnp.array([1.0, -1.0])
    dx = jnp.zeros_like(x0)
    assert jnp.allclose(geo.exp_map(x0, dx), x0)


def test_geodesic_endpoints(geo):
    x0 = jnp.array([-0.5, 0.2])
    x1 = jnp.array([0.4, -0.6])
    assert jnp.allclose(geo.geodesic(jnp.asarray(0.0), x0, x1), x0)
    assert jnp.allclose(geo.geodesic(jnp.asarray(1.0), x0, x1), x1)


def test_geodesic_stays_in_box_for_interior_endpoints(geo):
    x0 = jnp.array([-1.0, 1.0])
    x1 = jnp.array([1.0, -1.0])
    for t in jnp.linspace(0.0, 1.0, 11):
        p = geo.geodesic(t, x0, x1)
        assert jnp.all(p >= -1.0) and jnp.all(p <= 1.0)


def test_bounded_and_reflected_diverge_on_overflowing_step(geo):
    x0 = jnp.array([0.9])
    dx = jnp.array([0.5])
    clipped = Bounded().exp_map(x0, dx)
    reflected = geo.exp_map(x0, dx)
    assert not jnp.allclose(clipped, reflected)


@pytest.mark.parametrize("batch_shape", [(), (2, 3)])
def test_exp_map_leading_batch_dims_preserved(geo, batch_shape):
    D = 4
    x0 = jnp.zeros(batch_shape + (D,))
    dx = jnp.ones(batch_shape + (D,)) * 3.0
    out = geo.exp_map(x0, dx)
    assert out.shape == batch_shape + (D,)


def test_exp_map_formula_matches_source(geo):
    x0 = jnp.array([0.2, -0.6, 0.95])
    dx = jnp.array([2.7, -3.4, 0.02])
    y = x0 + dx
    folded = jnp.mod(y + 1.0, 4.0)
    expected = jnp.where(folded > 2.0, 4.0 - folded, folded) - 1.0
    assert jnp.allclose(geo.exp_map(x0, dx), expected)


def test_exp_map_landing_exactly_on_opposite_corner():
    geo = Reflected()
    x0 = jnp.array([1.0])
    dx = jnp.array([2.0])
    out = geo.exp_map(x0, dx)
    assert jnp.allclose(out, jnp.array([-1.0]))


def test_exp_map_landing_exactly_on_opposite_corner_negative_side():
    geo = Reflected()
    x0 = jnp.array([-1.0])
    dx = jnp.array([-2.0])
    out = geo.exp_map(x0, dx)
    assert jnp.allclose(out, jnp.array([1.0]))


def test_exp_map_tie_at_fold_boundary_is_continuous():
    # folded == 2.0 exactly is the tie point of the `folded > 2.0` branch;
    # both branches agree there so the fold must be continuous, not a jump.
    geo = Reflected()
    x0 = jnp.array([0.0])
    dx = jnp.array([1.0])  # y = 1.0 -> folded = mod(2.0, 4.0) = 2.0 exactly
    out = geo.exp_map(x0, dx)
    assert jnp.allclose(out, jnp.array([1.0]))


@pytest.mark.parametrize("batch_shape", [(4,)])
def test_exp_map_random_batch_matches_reference(geo, batch_shape):
    D = 3
    key = jax.random.key(1)
    k0, k1 = jax.random.split(key)
    x0 = jax.random.uniform(k0, batch_shape + (D,), minval=-1.0, maxval=1.0)
    dx = jax.random.uniform(k1, batch_shape + (D,), minval=-8.0, maxval=8.0)
    out = geo.exp_map(x0, dx)
    expected = reflect_into_box(x0 + dx)
    assert jnp.allclose(out, expected)
