"""Spherical chart: D hyperspherical angles -> Cartesian on sphere R^{D+1}."""

import jax.numpy as jnp
import pytest

from canna.charts import Spherical


def test_default_radius_is_one():
    chart = Spherical()
    assert jnp.allclose(chart.radius, 1.0)


def test_forward_shape_increments_last_axis():
    chart = Spherical()
    p = jnp.array([0.3, 1.2])
    out = chart.forward(p)
    assert out.shape == (3,)


def test_backward_shape_decrements_last_axis():
    chart = Spherical()
    x = jnp.array([1.0, 0.0, 0.0])
    out = chart.backward(x)
    assert out.shape == (2,)


def test_forward_output_has_correct_norm_d1():
    chart = Spherical(radius=1.0)
    p = jnp.array([0.7])
    out = chart.forward(p)
    assert jnp.allclose(jnp.linalg.norm(out), 1.0, atol=1e-4)


def test_forward_output_has_correct_norm_d2():
    chart = Spherical(radius=1.0)
    p = jnp.array([0.7, 2.1])
    out = chart.forward(p)
    assert jnp.allclose(jnp.linalg.norm(out), 1.0, atol=1e-4)


def test_radius_scales_forward_linearly():
    p = jnp.array([0.4, -1.3])
    unit = Spherical(radius=1.0).forward(p)
    scaled = Spherical(radius=3.0).forward(p)
    assert jnp.allclose(scaled, 3.0 * unit, atol=1e-4)


def test_forward_output_norm_equals_radius_general():
    chart = Spherical(radius=2.5)
    p = jnp.array([0.4, -1.3, 0.9])
    out = chart.forward(p)
    assert jnp.allclose(jnp.linalg.norm(out), 2.5, atol=1e-4)


def test_roundtrip_forward_backward_d1():
    chart = Spherical(radius=1.5)
    p = jnp.array([0.9])
    out = chart.backward(chart.forward(p))
    assert jnp.allclose(out, p, atol=1e-4)


def test_roundtrip_forward_backward_d2():
    chart = Spherical(radius=1.0)
    p = jnp.array([1.1, 0.4])
    out = chart.backward(chart.forward(p))
    assert jnp.allclose(out, p, atol=1e-4)


def test_d1_matches_standard_circle_parametrization():
    # D=1 hyperspherical chart should reduce to the ordinary (r*cos, r*sin) circle,
    # up to which axis holds cos vs sin -- so check via the sorted/abs invariant.
    chart = Spherical(radius=2.0)
    p = jnp.array([0.6])
    out = chart.forward(p)
    expected_components = jnp.sort(
        jnp.abs(jnp.array([2.0 * jnp.cos(0.6), 2.0 * jnp.sin(0.6)]))
    )
    assert jnp.allclose(jnp.sort(jnp.abs(out)), expected_components, atol=1e-4)


@pytest.mark.parametrize("batch_shape", [(), (4,), (2, 3)])
def test_forward_leading_batch_dims_preserved(batch_shape):
    D = 2
    chart = Spherical(radius=1.0)
    p = jnp.ones(batch_shape + (D,)) * 0.3
    out = chart.forward(p)
    assert out.shape == batch_shape + (D + 1,)


@pytest.mark.parametrize("batch_shape", [(), (4,)])
def test_roundtrip_leading_batch_dims(batch_shape):
    D = 2
    chart = Spherical(radius=1.0)
    p = jnp.broadcast_to(jnp.array([0.8, -0.2]), batch_shape + (D,))
    out = chart.backward(chart.forward(p))
    assert jnp.allclose(out, p, atol=1e-4)


def test_forward_matches_independent_reference_formula_d3():
    chart = Spherical(radius=2.0)
    p = jnp.array([0.3, 0.5, -0.4])  # azimuth, latitude1, latitude2
    out = chart.forward(p)
    az, lat1, lat2 = p
    expected = jnp.array(
        [
            2.0 * jnp.cos(lat2) * jnp.cos(lat1) * jnp.cos(az),
            2.0 * jnp.cos(lat2) * jnp.cos(lat1) * jnp.sin(az),
            2.0 * jnp.cos(lat2) * jnp.sin(lat1),
            2.0 * jnp.sin(lat2),
        ]
    )
    assert jnp.allclose(out, expected, atol=1e-4)


def test_forward_d1_exact_axis_order_is_cos_then_sin():
    # pins the exact axis order the recursion produces (azimuth is always axes [0,1]).
    chart = Spherical(radius=2.0)
    p = jnp.array([0.6])
    out = chart.forward(p)
    expected = jnp.array([2.0 * jnp.cos(0.6), 2.0 * jnp.sin(0.6)])
    assert jnp.allclose(out, expected, atol=1e-4)


def test_backward_azimuth_is_in_0_2pi_canonical_range():
    chart = Spherical(radius=1.0)
    p = jnp.array([-0.4, 0.1])
    recovered = chart.backward(chart.forward(p))
    assert recovered[0] >= 0.0
    assert recovered[0] < 2 * jnp.pi


def test_roundtrip_fails_outside_canonical_latitude_domain():
    # latitude=2.1 is outside [-pi/2, pi/2]: backward's arctan2(x[i],base_norm)
    # with base_norm>=0 can only recover values in [-pi/2, pi/2], so it cannot
    # invert this input -- backward is not a global inverse of forward, only a
    # local one on the conventional latitude domain.
    chart = Spherical(radius=1.0)
    p = jnp.array([0.7, 2.1])
    recovered = chart.backward(chart.forward(p))
    assert not jnp.allclose(recovered, p, atol=1e-3)
