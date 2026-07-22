"""Toroidal: product of circles in (cos,sin)-pair coords, geodesics wrap short way."""

import jax
import jax.numpy as jnp
import pytest

from canna.geometries import Toroidal
from canna.charts import Periodic


def circle_point(angle):
    return jnp.array([jnp.cos(angle), jnp.sin(angle)])


def test_log_map_zero_for_coincident_points():
    geo = Toroidal()
    x0 = circle_point(0.7)
    out = geo.log_map(x0, x0)
    assert jnp.allclose(out, 0.0, atol=1e-4)


def test_log_then_exp_recovers_x1_single_circle():
    geo = Toroidal()
    x0 = circle_point(0.3)
    x1 = circle_point(2.1)
    recovered = geo.exp_map(x0, geo.log_map(x0, x1))
    assert jnp.allclose(recovered, x1, atol=1e-4)


def test_geodesic_endpoints_single_circle():
    geo = Toroidal()
    x0 = circle_point(0.2)
    x1 = circle_point(2.5)
    assert jnp.allclose(geo.geodesic(jnp.asarray(0.0), x0, x1), x0, atol=1e-4)
    assert jnp.allclose(geo.geodesic(jnp.asarray(1.0), x0, x1), x1, atol=1e-4)


def test_geodesic_identity_when_x0_equals_x1():
    geo = Toroidal()
    x = circle_point(1.0)
    for t in jnp.array([0.0, 0.3, 0.5, 1.0]):
        assert jnp.allclose(geo.geodesic(t, x, x), x, atol=1e-4)


def test_geodesic_stays_on_circle_all_t():
    geo = Toroidal()
    x0 = circle_point(0.1)
    x1 = circle_point(4.0)
    for t in jnp.linspace(0.0, 1.0, 9):
        p = geo.geodesic(t, x0, x1)
        assert jnp.allclose(jnp.sum(p**2), 1.0, atol=1e-4)


def test_geodesic_takes_short_way_around_small_gap():
    # x0 at angle 0, x1 at angle -0.1 (equivalently 2pi-0.1): short way is
    # backward through 0, not forward through pi. Midpoint angle ~ -0.05.
    geo = Toroidal()
    x0 = circle_point(0.0)
    x1 = circle_point(-0.1)
    mid = geo.geodesic(jnp.asarray(0.5), x0, x1)
    mid_angle = jnp.arctan2(mid[1], mid[0])
    assert jnp.abs(mid_angle - (-0.05)) < 1e-3


def test_geodesic_short_way_across_zero_wrap():
    # x0 at angle 2pi - 0.1 (~ -0.1), x1 at angle 0.1: short way crosses the
    # 0/2pi seam directly, distance 0.2, not the long way (~2pi - 0.2).
    geo = Toroidal()
    x0 = circle_point(2 * jnp.pi - 0.1)
    x1 = circle_point(0.1)
    mid = geo.geodesic(jnp.asarray(0.5), x0, x1)
    mid_angle = jnp.arctan2(mid[1], mid[0])
    assert jnp.abs(mid_angle) < 1e-3


def test_product_of_two_independent_circles():
    geo = Toroidal()
    x0 = jnp.concatenate([circle_point(0.2), circle_point(1.0)])
    x1 = jnp.concatenate([circle_point(2.5), circle_point(1.0)])
    mid = geo.geodesic(jnp.asarray(0.5), x0, x1)
    # second circle unchanged (x0==x1 on that circle), stays fixed
    assert jnp.allclose(mid[2:], circle_point(1.0), atol=1e-4)
    assert jnp.allclose(jnp.sum(mid[:2] ** 2), 1.0, atol=1e-4)
    assert jnp.allclose(jnp.sum(mid[2:] ** 2), 1.0, atol=1e-4)


def test_antipodal_points_geodesic_endpoints_still_hold():
    # antipodal on a circle (angle diff = pi): direction ambiguous, but the
    # geodesic must still start/end exactly at x0/x1 and stay on the circle.
    geo = Toroidal()
    x0 = circle_point(0.0)
    x1 = circle_point(jnp.pi)
    assert jnp.allclose(geo.geodesic(jnp.asarray(0.0), x0, x1), x0, atol=1e-4)
    assert jnp.allclose(geo.geodesic(jnp.asarray(1.0), x0, x1), x1, atol=1e-4)
    mid = geo.geodesic(jnp.asarray(0.5), x0, x1)
    assert jnp.allclose(jnp.sum(mid**2), 1.0, atol=1e-4)


@pytest.mark.parametrize("batch_shape", [(), (5,)])
def test_geodesic_leading_batch_dims_preserved(batch_shape):
    geo = Toroidal()
    D2 = 4
    x0 = jnp.broadcast_to(
        jnp.concatenate([circle_point(0.1), circle_point(0.2)]), batch_shape + (D2,)
    )
    x1 = jnp.broadcast_to(
        jnp.concatenate([circle_point(1.1), circle_point(2.2)]), batch_shape + (D2,)
    )
    out = geo.geodesic(jnp.asarray(0.5), x0, x1)
    assert out.shape == batch_shape + (D2,)


def test_toroidal_geodesic_consistent_with_periodic_chart_roundtrip():
    # chart.forward(angle) -> geometry point -> geodesic
    # midpoint should map back (via chart.backward) to the arithmetic-mean
    # angle for the simple non-wrapping case.
    chart = Periodic(period=jnp.array([2 * jnp.pi]))
    geo = Toroidal()
    a0, a1 = 0.4, 1.0
    x0, x1 = chart.forward(jnp.array([a0])), chart.forward(jnp.array([a1]))
    mid = geo.geodesic(jnp.asarray(0.5), x0, x1)
    recovered_angle = chart.backward(mid)
    assert jnp.allclose(recovered_angle, jnp.array([(a0 + a1) / 2]), atol=1e-3)


def test_log_map_returns_ambient_tangent_one_pair_per_circle():
    # the tangent is ambient: a (cos,sin)-pair velocity per circle, same
    # dimension as the point.
    geo = Toroidal()
    x0 = jnp.concatenate([circle_point(0.1), circle_point(0.2)])
    x1 = jnp.concatenate([circle_point(1.0), circle_point(2.0)])
    tangent = geo.log_map(x0, x1)
    assert tangent.shape == (4,)


def test_exp_map_zero_tangent_is_identity():
    geo = Toroidal()
    x0 = circle_point(1.2)
    dx = jnp.zeros(2)
    out = geo.exp_map(x0, dx)
    assert jnp.allclose(out, x0, atol=1e-4)


def test_exp_map_ambient_dx_stays_on_unit_circle():
    geo = Toroidal()
    x0 = circle_point(0.5)
    dx = jnp.array([0.9, -0.3])
    out = geo.exp_map(x0, dx)
    assert jnp.allclose(jnp.sum(out**2), 1.0, atol=1e-4)


def test_log_map_wraps_into_minus_pi_pi_not_shortest_unsigned():
    # dphi=3.0 (just under pi): arc stays ~3.0 (no wrap needed); dphi=4.0 (>pi)
    # must wrap to 4.0-2*pi (~ -2.283), the short way, not stay at +4.0.
    geo = Toroidal()
    x0 = circle_point(0.0)
    x1_no_wrap = circle_point(3.0)
    x1_wrap = circle_point(4.0)
    # at phi0=0 the d/dphi direction is (0,1), so the arc reads off component 1
    tangent_no_wrap = geo.log_map(x0, x1_no_wrap)
    assert jnp.allclose(tangent_no_wrap[1], 3.0, atol=1e-3)
    tangent_wrap = geo.log_map(x0, x1_wrap)
    assert jnp.allclose(tangent_wrap[1], 4.0 - 2 * jnp.pi, atol=1e-3)


def test_log_map_shape_matches_ambient_point_shape():
    geo = Toroidal()
    x0 = jnp.concatenate([circle_point(0.1), circle_point(0.2)])
    x1 = jnp.concatenate([circle_point(1.0), circle_point(2.0)])
    tangent = geo.log_map(x0, x1)
    assert tangent.shape == (4,)


def test_exp_map_dx_shape_matches_ambient_point_shape():
    geo = Toroidal()
    x0 = jnp.concatenate([circle_point(0.1), circle_point(0.2)])
    dx = jnp.array([0.3, -0.1, 0.2, 0.4])
    out = geo.exp_map(x0, dx)
    assert out.shape == (4,)


def test_log_map_dphi_exactly_pi_wraps_to_negative_pi():
    # exact antipodal angles (not routed through circle_point's trig, which
    # would leave a float-epsilon residue on the pi boundary)
    geo = Toroidal()
    x0 = jnp.array([1.0, 0.0])
    x1 = jnp.array([-1.0, 0.0])
    tangent = geo.log_map(x0, x1)
    # d/dphi at phi0=0 is (0,1), so the -pi arc lands entirely on component 1
    assert jnp.allclose(tangent, jnp.array([0.0, -jnp.pi]), atol=1e-6)


def test_log_map_per_circle_independence():
    # changing only the second circle's angle must not perturb the first
    # circle's arc component
    geo = Toroidal()
    x0 = jnp.concatenate([circle_point(0.3), circle_point(0.5)])
    x1 = jnp.concatenate([circle_point(0.3), circle_point(2.0)])
    tangent = geo.log_map(x0, x1)
    assert jnp.allclose(tangent[:2], 0.0, atol=1e-4)
    assert jnp.allclose(jnp.linalg.norm(tangent[2:]), 1.5, atol=1e-3)


def test_exp_map_dx_pi_reaches_antipode_on_circle():
    geo = Toroidal()
    x0 = circle_point(0.4)
    dx = jnp.pi * jnp.array([-jnp.sin(0.4), jnp.cos(0.4)])
    out = geo.exp_map(x0, dx)
    assert jnp.allclose(out, -x0, atol=1e-4)


def test_exp_map_periodic_in_dx_with_period_2pi():
    geo = Toroidal()
    x0 = circle_point(0.4)
    d_dphi = jnp.array([-jnp.sin(0.4), jnp.cos(0.4)])
    out = geo.exp_map(x0, 0.9 * d_dphi)
    out_plus_full_turn = geo.exp_map(x0, (0.9 + 2 * jnp.pi) * d_dphi)
    assert jnp.allclose(out, out_plus_full_turn, atol=1e-4)


def test_exp_map_arbitrarily_large_dx_stays_on_unit_circle():
    geo = Toroidal()
    x0 = circle_point(0.2)
    dx = jnp.array([37.0, -12.0])
    out = geo.exp_map(x0, dx)
    assert jnp.allclose(jnp.sum(out**2), 1.0, atol=1e-4)


def test_geodesic_jacobian_wrt_t_finite_generic_points():
    # mirrors problems.py: jax.jacobian(geodesic) w.r.t. t must stay finite
    geo = Toroidal()
    x0 = circle_point(0.1)
    x1 = circle_point(2.3)
    jac = jax.jacobian(geo.geodesic)(0.4, x0, x1)
    assert jnp.all(jnp.isfinite(jac))


def test_geodesic_consistent_with_periodic_chart_roundtrip_across_wrap():
    # extends the existing non-wrapping roundtrip check to the case where the
    # short way crosses the 0/2pi seam
    chart = Periodic(period=jnp.array([2 * jnp.pi]))
    geo = Toroidal()
    a0, a1 = 2 * jnp.pi - 0.1, 0.1
    x0, x1 = chart.forward(jnp.array([a0])), chart.forward(jnp.array([a1]))
    mid = geo.geodesic(jnp.asarray(0.5), x0, x1)
    recovered_angle = chart.backward(mid)
    # angles compare modulo the period: 0 and 2pi are the same point
    offset = jnp.mod(recovered_angle - 0.0 + jnp.pi, 2 * jnp.pi) - jnp.pi
    assert jnp.allclose(offset, 0.0, atol=1e-3)


def test_antipodal_log_map_stays_finite():
    # no division singularity at angle-diff==pi: the arc is a jnp.mod wrap, so
    # the branch is picked by the mod convention and the result stays finite.
    geo = Toroidal()
    x0 = circle_point(0.0)
    x1 = circle_point(jnp.pi)
    out = geo.log_map(x0, x1)
    assert jnp.all(jnp.isfinite(out))


def test_geodesic_jacobian_wrt_t_finite_at_exact_antipode():
    # mirrors problems.py's jax.jacobian call, at the dphi==pi cut locus where
    # the mod-wrap could in principle make the arc's dependence ill-behaved
    geo = Toroidal()
    x0 = circle_point(0.0)
    x1 = circle_point(jnp.pi)
    jac = jax.jacobian(geo.geodesic)(0.5, x0, x1)
    assert jnp.all(jnp.isfinite(jac))
    assert jac.shape == (2,)


def test_geodesic_jacobian_wrt_t_finite_when_base_lands_on_target():
    # train_sample draws dx = jax.jacobian(geodesic, t); when the base draw x0
    # coincides with the target x1 the log_map runs 0/0 under AD, but the arc
    # length is identically zero so the velocity limit is a finite zero vector.
    geo = Toroidal()
    x = circle_point(0.7)
    jac = jax.jacobian(geo.geodesic)(0.4, x, x)
    assert jnp.all(jnp.isfinite(jac))
    assert jnp.allclose(jac, 0.0, atol=1e-4)


def test_geodesic_jacobian_wrt_t_shape_matches_point_multi_circle():
    geo = Toroidal()
    x0 = jnp.concatenate([circle_point(0.1), circle_point(0.2)])
    x1 = jnp.concatenate([circle_point(1.0), circle_point(2.0)])
    jac = jax.jacobian(geo.geodesic)(0.4, x0, x1)
    assert jac.shape == (4,)
    assert jnp.all(jnp.isfinite(jac))


def test_exp_map_purely_radial_dx_is_identity():
    # dx parallel to x0 itself (no d/dphi component) must project to zero arc
    geo = Toroidal()
    x0 = circle_point(0.6)
    dx_radial = 2.5 * x0
    out = geo.exp_map(x0, dx_radial)
    assert jnp.allclose(out, x0, atol=1e-4)


def test_exp_map_zero_dx_does_not_renormalize_nonunit_x0():
    # exp_map's d/dphi projection reads the angle as phi0 = arctan2(x0), so a
    # zero tangent must return a non-unit-norm x0 unchanged rather than snapping
    # it back onto the unit circle.
    geo = Toroidal()
    x0 = 2.0 * circle_point(0.6)
    out = geo.exp_map(x0, jnp.zeros(2))
    assert jnp.allclose(out, x0, atol=1e-4)


def test_batched_log_map_mixes_coincident_generic_and_antipodal_rows():
    # one call, three independent rows at very different dphi regimes: each
    # row's arc must be computed independently of the others
    geo = Toroidal()
    x0 = jnp.stack([circle_point(0.0), circle_point(0.5), circle_point(0.0)])
    x1 = jnp.stack([circle_point(0.0), circle_point(1.2), circle_point(jnp.pi)])
    tangent = geo.log_map(x0, x1)
    assert tangent.shape == (3, 2)
    assert jnp.allclose(tangent[0], 0.0, atol=1e-4)
    assert jnp.allclose(jnp.linalg.norm(tangent[1]), 0.7, atol=1e-3)
    assert jnp.allclose(jnp.linalg.norm(tangent[2]), jnp.pi, atol=1e-3)
    assert jnp.all(jnp.isfinite(tangent))


def test_exp_map_preserves_radius_non_unit_circle():
    geo = Toroidal()
    radius = 2.5
    x0 = radius * circle_point(0.6)
    dx = jnp.array([0.4, -0.9])
    out = geo.exp_map(x0, dx)
    assert jnp.allclose(jnp.linalg.norm(out), radius, atol=1e-4)


def test_log_then_exp_recovers_x1_non_unit_radius():
    geo = Toroidal()
    radius = 3.0
    x0 = radius * circle_point(0.3)
    x1 = radius * circle_point(2.1)
    recovered = geo.exp_map(x0, geo.log_map(x0, x1))
    assert jnp.allclose(recovered, x1, atol=1e-4)


def test_log_map_scales_with_radius():
    # the tangent is an arc length, so the same angular separation on a bigger
    # circle gives a proportionally bigger tangent
    geo = Toroidal()
    small = geo.log_map(circle_point(0.2), circle_point(1.4))
    big = geo.log_map(5.0 * circle_point(0.2), 5.0 * circle_point(1.4))
    assert jnp.allclose(big, 5.0 * small, atol=1e-4)


def test_exp_map_arc_length_equals_tangent_norm_any_radius():
    geo = Toroidal()
    for radius in (1.0, 4.0):
        x0 = radius * circle_point(0.7)
        dx = jnp.array([0.3, -0.5])
        out = geo.exp_map(x0, dx)
        cos_angle = jnp.dot(x0, out) / radius**2
        arc = radius * jnp.arccos(jnp.clip(cos_angle, -1.0, 1.0))
        # dx's radial part is discarded, so compare against its d/dphi component
        d_dphi = jnp.array([-jnp.sin(0.7), jnp.cos(0.7)])
        assert jnp.allclose(arc, jnp.abs(jnp.dot(dx, d_dphi)), atol=1e-4)


def test_independent_radii_per_circle():
    # each circle carries its own radius; neither is normalized to the other
    geo = Toroidal()
    x0 = jnp.concatenate([1.0 * circle_point(0.2), 3.0 * circle_point(1.0)])
    x1 = jnp.concatenate([1.0 * circle_point(2.5), 3.0 * circle_point(2.2)])
    mid = geo.geodesic(jnp.asarray(0.5), x0, x1)
    assert jnp.allclose(jnp.linalg.norm(mid[:2]), 1.0, atol=1e-4)
    assert jnp.allclose(jnp.linalg.norm(mid[2:]), 3.0, atol=1e-4)


def test_geodesic_endpoints_non_unit_radius():
    geo = Toroidal()
    x0 = 2.0 * circle_point(0.2)
    x1 = 2.0 * circle_point(2.5)
    assert jnp.allclose(geo.geodesic(jnp.asarray(0.0), x0, x1), x0, atol=1e-4)
    assert jnp.allclose(geo.geodesic(jnp.asarray(1.0), x0, x1), x1, atol=1e-4)


def test_log_map_independent_of_x1_radius():
    # x1 supplies only an angle; its radius is never read
    geo = Toroidal()
    x0 = 2.0 * circle_point(0.3)
    x1 = circle_point(1.9)
    assert jnp.allclose(geo.log_map(x0, x1), geo.log_map(x0, 7.0 * x1), atol=1e-6)


def test_geodesic_keeps_x0_radius_when_x1_radius_differs():
    geo = Toroidal()
    x0 = 2.0 * circle_point(0.3)
    x1 = 5.0 * circle_point(1.9)
    out = geo.geodesic(jnp.asarray(1.0), x0, x1)
    assert jnp.allclose(jnp.linalg.norm(out), 2.0, atol=1e-6)
    assert jnp.allclose(jnp.arctan2(out[1], out[0]), 1.9, atol=1e-6)
