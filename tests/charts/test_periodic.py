"""Periodic: angle -> (cos,sin) pair per D, period-scaled, D->D*2 roundtrip."""

import jax.numpy as jnp
import pytest

from canna.charts import Periodic


def test_default_period_is_two_pi_len_one():
    chart = Periodic()
    assert chart.period.shape == (1,)
    assert jnp.allclose(chart.period, 2 * jnp.pi)


def test_forward_shape_doubles_last_axis():
    chart = Periodic(period=jnp.array([2 * jnp.pi, 4.0, 1.0]))
    p = jnp.array([0.3, 1.0, 0.5])
    out = chart.forward(p)
    assert out.shape == (6,)


def test_backward_shape_halves_last_axis():
    chart = Periodic(period=jnp.array([2 * jnp.pi, 4.0]))
    x = jnp.array([1.0, 0.0, 0.0, 1.0])
    out = chart.backward(x)
    assert out.shape == (2,)


def test_forward_pairs_are_unit_norm():
    # cos^2+sin^2==1 per angle; layout of the D*2 axis is unspecified, so check
    # via total sum-of-squares == D (robust to interleaved vs blocked layout).
    chart = Periodic(period=jnp.array([2 * jnp.pi, 3.0, 5.0]))
    p = jnp.array([0.3, 1.1, 4.2])
    out = chart.forward(p)
    assert jnp.allclose(jnp.sum(out**2), 3.0, atol=1e-4)


def test_roundtrip_forward_backward_within_range():
    chart = Periodic(period=jnp.array([2 * jnp.pi, 4.0]))
    p = jnp.array([1.2, 2.5])
    out = chart.backward(chart.forward(p))
    assert jnp.allclose(out, p, atol=1e-4)


def test_periodicity_forward_invariant_under_full_period_shift():
    chart = Periodic(period=jnp.array([2 * jnp.pi, 4.0]))
    p = jnp.array([1.0, 3.0])
    p_shifted = p + chart.period
    assert jnp.allclose(chart.forward(p), chart.forward(p_shifted), atol=1e-4)


def test_zero_angle_maps_to_pair_containing_one_and_zero():
    # cos(0)=1, sin(0)=0 for every angle, regardless of cos/sin axis ordering
    chart = Periodic(period=jnp.array([2 * jnp.pi, 5.0, 1.0]))
    p = jnp.zeros(3)
    out = chart.forward(p)
    sorted_vals = jnp.sort(out.reshape(3, 2), axis=-1)
    assert jnp.allclose(sorted_vals, jnp.array([0.0, 1.0]), atol=1e-4)


def test_quarter_period_maps_to_pair_zero_one_up_to_sign():
    # angle = period/4 -> phase pi/2 -> {cos,sin} = {0, +-1}
    chart = Periodic(period=jnp.array([4.0]))
    p = jnp.array([1.0])
    out = chart.forward(p)
    assert jnp.allclose(jnp.sort(jnp.abs(out)), jnp.array([0.0, 1.0]), atol=1e-4)


@pytest.mark.parametrize("batch_shape", [(), (5,), (2, 3)])
def test_forward_leading_batch_dims_preserved(batch_shape):
    D = 3
    chart = Periodic(period=jnp.array([2 * jnp.pi, 4.0, 1.0]))
    p = jnp.ones(batch_shape + (D,)) * 0.4
    out = chart.forward(p)
    assert out.shape == batch_shape + (2 * D,)


@pytest.mark.parametrize("batch_shape", [(), (5,), (2, 3)])
def test_roundtrip_leading_batch_dims(batch_shape):
    D = 2
    chart = Periodic(period=jnp.array([2 * jnp.pi, 4.0]))
    p = jnp.broadcast_to(jnp.array([1.0, 2.0]), batch_shape + (D,))
    out = chart.backward(chart.forward(p))
    assert jnp.allclose(out, p, atol=1e-4)


def test_negative_angle_wraps_into_canonical_range():
    # backward returns an angle equivalent mod period; check it's the same point
    # on the circle as forward(-eps) round-tripped, not a specific numeric range.
    chart = Periodic(period=jnp.array([2 * jnp.pi]))
    p_neg = jnp.array([-0.3])
    recovered = chart.backward(chart.forward(p_neg))
    assert jnp.allclose(jnp.cos(recovered), jnp.cos(p_neg), atol=1e-4)
    assert jnp.allclose(jnp.sin(recovered), jnp.sin(p_neg), atol=1e-4)


def test_forward_layout_is_interleaved_cos_sin_per_angle():
    chart = Periodic(period=jnp.array([2 * jnp.pi, 4.0]))
    p = jnp.array([0.7, 1.3])
    out = chart.forward(p)
    angle = p / chart.period * 2 * jnp.pi
    expected = jnp.stack(
        [jnp.cos(angle[0]), jnp.sin(angle[0]), jnp.cos(angle[1]), jnp.sin(angle[1])]
    )
    assert jnp.allclose(out, expected, atol=1e-4)


def test_backward_canonicalizes_to_0_period_exact_value():
    chart = Periodic(period=jnp.array([2 * jnp.pi]))
    p_neg = jnp.array([-0.3])
    recovered = chart.backward(chart.forward(p_neg))
    assert jnp.allclose(recovered, jnp.array([2 * jnp.pi - 0.3]), atol=1e-4)


def test_backward_never_returns_negative_or_full_period():
    chart = Periodic(period=jnp.array([2 * jnp.pi, 5.0]))
    p = jnp.array([-10.0, 17.0])
    recovered = chart.backward(chart.forward(p))
    assert jnp.all(recovered >= 0.0)
    assert jnp.all(recovered < chart.period)
