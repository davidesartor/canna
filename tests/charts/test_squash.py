"""Squash: p in [low, high] <-> x in R via arctanh, forward/backward invertibility."""

import jax
import jax.numpy as jnp
import pytest

from canna.charts import Squash


def test_default_box_is_minus_one_to_one():
    chart = Squash()
    assert chart.physical_dim == 1 and chart.flow_dim == 1


def test_midpoint_maps_to_zero_default_box():
    chart = Squash()
    assert jnp.allclose(chart.forward(jnp.array([0.0])), 0.0, atol=1e-6)


def test_midpoint_maps_to_zero_custom_box():
    low, high = jnp.array([2.0, -4.0]), jnp.array([6.0, 4.0])
    chart = Squash(low=low, high=high)
    mid = (low + high) / 2
    assert jnp.allclose(chart.forward(mid), 0.0, atol=1e-6)


def test_forward_matches_arctanh_formula():
    low, high = jnp.array([0.0, -1.0]), jnp.array([4.0, 3.0])
    chart = Squash(low=low, high=high)
    p = jnp.array([1.0, 2.0])
    expected = jnp.arctanh(2 * (p - low) / (high - low) - 1)
    assert jnp.allclose(chart.forward(p), expected, atol=1e-6)


def test_backward_matches_tanh_formula():
    low, high = jnp.array([0.0, -1.0]), jnp.array([4.0, 3.0])
    chart = Squash(low=low, high=high)
    x = jnp.array([0.7, -1.3])
    expected = low + (high - low) * (jnp.tanh(x) + 1) / 2
    assert jnp.allclose(chart.backward(x), expected, atol=1e-6)


def test_roundtrip_interior_forward_backward():
    low, high = jnp.array([0.0, -2.0, 5.0]), jnp.array([1.0, 2.0, 9.0])
    chart = Squash(low=low, high=high)
    p = jnp.array([0.3, 0.5, 6.4])
    assert jnp.allclose(chart.backward(chart.forward(p)), p, atol=1e-5)


def test_roundtrip_backward_forward():
    low, high = jnp.array([0.0, -2.0]), jnp.array([1.0, 2.0])
    chart = Squash(low=low, high=high)
    x = jnp.array([1.2, -3.1])
    assert jnp.allclose(chart.forward(chart.backward(x)), x, atol=1e-5)


def test_backward_stays_in_the_closed_box_and_is_finite_for_extreme_coords():
    low, high = jnp.array([-3.0]), jnp.array([7.0])
    chart = Squash(low=low, high=high)
    x = jnp.array([-1e3, -50.0, 0.0, 50.0, 1e3])
    p = jax.vmap(chart.backward)(x[:, None])[:, 0]
    assert bool(jnp.all(p >= low)) and bool(jnp.all(p <= high))
    assert bool(jnp.all(jnp.isfinite(p)))


def test_backward_saturates_to_the_boundary_when_tanh_hits_plus_minus_one():
    low, high = jnp.array([-3.0]), jnp.array([7.0])
    chart = Squash(low=low, high=high)
    # float tanh saturates to +-1 for large |x|, so backward reaches the edge exactly
    assert chart.backward(jnp.array([-1e3]))[0] == low[0]
    assert chart.backward(jnp.array([1e3]))[0] == high[0]


def test_backward_is_strictly_interior_for_moderate_coords():
    low, high = jnp.array([-3.0]), jnp.array([7.0])
    chart = Squash(low=low, high=high)
    x = jnp.array([-5.0, -1.0, 0.0, 1.0, 5.0])
    p = jax.vmap(chart.backward)(x[:, None])[:, 0]
    assert bool(jnp.all(p > low)) and bool(jnp.all(p < high))


def test_forward_diverges_at_lower_edge():
    chart = Squash()
    assert chart.forward(jnp.array([-1.0]))[0] == -jnp.inf


def test_forward_diverges_at_upper_edge():
    chart = Squash()
    assert chart.forward(jnp.array([1.0]))[0] == jnp.inf


def test_forward_outside_box_is_nan_and_unguarded():
    chart = Squash()
    assert bool(jnp.all(jnp.isnan(chart.forward(jnp.array([1.5, -2.0])))))


def test_forward_is_strictly_monotone_increasing():
    chart = Squash()
    grid = jnp.linspace(-0.99, 0.99, 50)[:, None]
    x = jax.vmap(chart.forward)(grid)[:, 0]
    assert bool(jnp.all(jnp.diff(x) > 0))


def test_forward_gradient_blows_up_toward_the_edge():
    chart = Squash()
    grad = jax.grad(lambda p: chart.forward(p)[0])
    near_center = grad(jnp.array([0.0]))[0]
    near_edge = grad(jnp.array([0.99]))[0]
    assert near_edge > near_center > 0
    assert near_edge > 10.0


@pytest.mark.parametrize("batch_shape", [(), (4,), (2, 3)])
def test_forward_preserves_batch_shape(batch_shape):
    D = 3
    chart = Squash(low=jnp.zeros(D), high=jnp.ones(D))
    p = jnp.full(batch_shape + (D,), 0.5)
    assert chart.forward(p).shape == batch_shape + (D,)


@pytest.mark.parametrize("batch_shape", [(), (4,), (2, 3)])
def test_roundtrip_preserves_batch_shape(batch_shape):
    D = 2
    low, high = jnp.array([0.0, -1.0]), jnp.array([2.0, 1.0])
    chart = Squash(low=low, high=high)
    p = jnp.broadcast_to(jnp.array([0.4, 0.2]), batch_shape + (D,))
    out = chart.backward(chart.forward(p))
    assert out.shape == batch_shape + (D,)
    assert jnp.allclose(out, p, atol=1e-5)


def test_physical_dim_equals_flow_dim():
    chart = Squash(low=jnp.zeros(4), high=jnp.ones(4))
    assert chart.physical_dim == chart.flow_dim == 4
