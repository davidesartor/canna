"""Spherical geometry: great-circle arcs on a sphere of any radius, embedded coords."""

import jax
import jax.numpy as jnp
import pytest

from canna.geometries import Spherical
from canna.charts import Spherical as SphericalChart


def test_log_map_zero_for_coincident_points():
    geo = Spherical()
    x0 = jnp.array([1.0, 0.0, 0.0])
    out = geo.log_map(x0, x0)
    assert jnp.allclose(out, 0.0, atol=1e-4)


def test_exp_map_zero_tangent_is_identity():
    geo = Spherical()
    x0 = jnp.array([0.0, 1.0, 0.0])
    dx = jnp.zeros(3)
    out = geo.exp_map(x0, dx)
    assert jnp.allclose(out, x0, atol=1e-4)


def test_log_map_tangent_is_ambient_and_orthogonal_to_x0():
    geo = Spherical()
    x0 = jnp.array([1.0, 0.0, 0.0])
    x1 = jnp.array([0.0, 1.0, 0.0])
    tangent = geo.log_map(x0, x1)
    assert tangent.shape == (3,)
    assert jnp.abs(jnp.dot(tangent, x0)) < 1e-4


def test_log_then_exp_recovers_x1():
    geo = Spherical()
    x0 = jnp.array([1.0, 0.0, 0.0])
    x1 = jnp.array([0.0, 1.0, 0.0])
    recovered = geo.exp_map(x0, geo.log_map(x0, x1))
    assert jnp.allclose(recovered, x1, atol=1e-4)


def test_log_then_exp_recovers_x1_nonorthogonal():
    geo = Spherical()
    x0 = jnp.array([1.0, 0.0, 0.0])
    x1 = jnp.array([0.6, 0.8, 0.0])
    recovered = geo.exp_map(x0, geo.log_map(x0, x1))
    assert jnp.allclose(recovered, x1, atol=1e-4)


def test_exp_map_preserves_norm_unit_sphere():
    geo = Spherical()
    x0 = jnp.array([1.0, 0.0, 0.0])
    dx = jnp.array([0.0, 0.5, 0.0])
    out = geo.exp_map(x0, dx)
    assert jnp.allclose(jnp.linalg.norm(out), 1.0, atol=1e-4)


def test_exp_map_preserves_norm_arbitrary_radius():
    geo = Spherical()
    radius = 3.0
    x0 = jnp.array([radius, 0.0, 0.0])
    dx = jnp.array([0.0, 1.0, 0.0])
    out = geo.exp_map(x0, dx)
    assert jnp.allclose(jnp.linalg.norm(out), radius, atol=1e-4)


def test_geodesic_endpoints():
    geo = Spherical()
    x0 = jnp.array([1.0, 0.0, 0.0])
    x1 = jnp.array([0.0, 1.0, 0.0])
    assert jnp.allclose(geo.geodesic(jnp.asarray(0.0), x0, x1), x0, atol=1e-4)
    assert jnp.allclose(geo.geodesic(jnp.asarray(1.0), x0, x1), x1, atol=1e-4)


def test_geodesic_identity_when_x0_equals_x1():
    geo = Spherical()
    x = jnp.array([0.0, 0.0, 2.0])
    for t in jnp.array([0.0, 0.25, 0.5, 1.0]):
        assert jnp.allclose(geo.geodesic(t, x, x), x, atol=1e-4)


def test_geodesic_stays_on_sphere_all_t():
    geo = Spherical()
    radius = 2.0
    x0 = jnp.array([radius, 0.0, 0.0])
    x1 = jnp.array([0.0, radius, 0.0])
    for t in jnp.linspace(0.0, 1.0, 9):
        p = geo.geodesic(t, x0, x1)
        assert jnp.allclose(jnp.linalg.norm(p), radius, atol=1e-4)


def test_great_circle_midpoint_equidistant_from_endpoints():
    geo = Spherical()
    x0 = jnp.array([1.0, 0.0, 0.0])
    x1 = jnp.array([0.0, 1.0, 0.0])
    mid = geo.geodesic(jnp.asarray(0.5), x0, x1)
    d0 = jnp.linalg.norm(mid - x0)
    d1 = jnp.linalg.norm(mid - x1)
    assert jnp.allclose(d0, d1, atol=1e-4)


def test_great_circle_orthogonal_quarter_turn_midpoint_direction():
    # x0=(1,0,0), x1=(0,1,0) on unit sphere: geodesic midpoint should lie
    # along the symmetric direction (1,1,0)/sqrt(2).
    geo = Spherical()
    x0 = jnp.array([1.0, 0.0, 0.0])
    x1 = jnp.array([0.0, 1.0, 0.0])
    mid = geo.geodesic(jnp.asarray(0.5), x0, x1)
    expected = jnp.array([1.0, 1.0, 0.0]) / jnp.sqrt(2.0)
    assert jnp.allclose(mid, expected, atol=1e-4)


def test_antipodal_points_direction_undefined_but_endpoints_hold():
    # antipodal points: infinitely many great circles connect them, direction
    # ambiguous/unstable -- only assert the invariant that must hold regardless.
    geo = Spherical()
    x0 = jnp.array([1.0, 0.0, 0.0])
    x1 = jnp.array([-1.0, 0.0, 0.0])
    assert jnp.allclose(geo.geodesic(jnp.asarray(0.0), x0, x1), x0, atol=1e-4)
    assert jnp.allclose(geo.geodesic(jnp.asarray(1.0), x0, x1), x1, atol=1e-4)
    mid = geo.geodesic(jnp.asarray(0.5), x0, x1)
    assert jnp.allclose(jnp.linalg.norm(mid), 1.0, atol=1e-4)


@pytest.mark.parametrize("batch_shape", [(), (5,)])
def test_geodesic_leading_batch_dims_preserved(batch_shape):
    geo = Spherical()
    D = 3
    x0 = jnp.broadcast_to(jnp.array([1.0, 0.0, 0.0]), batch_shape + (D,))
    x1 = jnp.broadcast_to(jnp.array([0.0, 1.0, 0.0]), batch_shape + (D,))
    out = geo.geodesic(jnp.asarray(0.5), x0, x1)
    assert out.shape == batch_shape + (D,)


def test_log_map_scales_with_radius_not_fixed_unit():
    # log_map's magnitude is |x0|*angle -- a bigger sphere means a bigger
    # tangent for the same angular separation, since it isn't normalized by a
    # separate "radius" parameter (there is none on the geometry) but purely
    # derived from |x0|.
    geo = Spherical()
    x0_unit = jnp.array([1.0, 0.0, 0.0])
    x1_unit = jnp.array([0.0, 1.0, 0.0])
    x0_big = jnp.array([5.0, 0.0, 0.0])
    x1_big = jnp.array([0.0, 5.0, 0.0])
    small = geo.log_map(x0_unit, x1_unit)
    big = geo.log_map(x0_big, x1_big)
    assert jnp.allclose(big, 5.0 * small, atol=1e-4)


def test_exp_map_arc_length_equals_tangent_norm():
    # the ambient tangent's norm is the geodesic arc length travelled,
    # whatever the sphere's radius.
    geo = Spherical()
    for radius in (1.0, 3.0):
        x0 = jnp.array([radius, 0.0, 0.0])
        dx = jnp.array([0.0, -0.5, 1.2])
        out = geo.exp_map(x0, dx)
        cos_angle = jnp.dot(x0, out) / radius**2
        arc = radius * jnp.arccos(jnp.clip(cos_angle, -1.0, 1.0))
        assert jnp.allclose(arc, jnp.linalg.norm(dx), atol=1e-4)


def test_antipodal_log_map_picks_a_direction_and_reaches_x1():
    # for exact antipodes x1 - cos_angle*x0 is the exact zero vector: no
    # direction survives, so log_map falls back to an arbitrary tangent axis.
    # a pi rotation lands on -x0 whichever direction is used, so the endpoint
    # is still exact even though the path is arbitrary.
    geo = Spherical()
    x0 = jnp.array([1.0, 0.0, 0.0])
    x1 = jnp.array([-1.0, 0.0, 0.0])
    tangent = geo.log_map(x0, x1)
    assert jnp.allclose(jnp.linalg.norm(tangent), jnp.pi, atol=1e-4)
    assert jnp.allclose(geo.geodesic(jnp.asarray(1.0), x0, x1), x1, atol=1e-4)


def test_antipodal_reached_from_non_axis_aligned_x0():
    geo = Spherical()
    x0 = jnp.array([0.6, 0.8, 0.0])
    x1 = -x0
    assert jnp.allclose(geo.geodesic(jnp.asarray(1.0), x0, x1), x1, atol=1e-4)
    mid = geo.geodesic(jnp.asarray(0.5), x0, x1)
    assert jnp.allclose(jnp.linalg.norm(mid), 1.0, atol=1e-4)


def test_log_map_tangent_orthogonal_to_x0_higher_dimension():
    # the tangent is ambient: it must be orthogonal to x0, i.e. a genuine
    # tangent-plane vector, for a generic non-aligned x0
    geo = Spherical()
    x0 = jnp.array([1.0, 2.0, -1.0, 0.5])
    x1 = jnp.array([-2.0, 1.0, 0.5, 1.0])
    tangent = geo.log_map(x0, x1)
    assert jnp.abs(jnp.dot(tangent, x0)) < 1e-4


def test_antipodal_endpoints_hold_in_higher_dimension():
    geo = Spherical()
    x0 = jnp.array([1.0, 2.0, -1.0, 0.5])
    x1 = -x0
    assert jnp.allclose(geo.geodesic(jnp.asarray(0.0), x0, x1), x0, atol=1e-4)
    assert jnp.allclose(geo.geodesic(jnp.asarray(1.0), x0, x1), x1, atol=1e-4)


def test_exp_map_full_turn_returns_to_start():
    geo = Spherical()
    x0 = jnp.array([1.0, 0.0, 0.0])
    direction = jnp.array([0.0, 1.0, 0.0])
    dx = 2 * jnp.pi * direction
    out = geo.exp_map(x0, dx)
    assert jnp.allclose(out, x0, atol=1e-4)


def test_exp_map_half_turn_reaches_antipode():
    geo = Spherical()
    x0 = jnp.array([1.0, 0.0, 0.0])
    direction = jnp.array([0.0, 1.0, 0.0])
    dx = jnp.pi * direction
    out = geo.exp_map(x0, dx)
    assert jnp.allclose(out, -x0, atol=1e-4)


def test_exp_map_preserves_norm_for_large_dx_beyond_half_turn():
    geo = Spherical()
    x0 = jnp.array([2.0, 0.0, 0.0])
    dx = jnp.array([0.0, 10.0, 3.0])
    out = geo.exp_map(x0, dx)
    assert jnp.allclose(jnp.linalg.norm(out), 2.0, atol=1e-4)


def test_geodesic_jacobian_wrt_t_finite_generic_points():
    # mirrors problems.py: jax.jacobian(geodesic) w.r.t. t must stay finite
    # for a generic (non cut-locus) pair of points
    geo = Spherical()
    x0 = jnp.array([1.0, 0.0, 0.0])
    x1 = jnp.array([0.6, 0.8, 0.0])
    jac = jax.jacobian(geo.geodesic)(0.3, x0, x1)
    assert jnp.all(jnp.isfinite(jac))


def test_geodesic_jacobian_wrt_t_finite_when_base_lands_on_target():
    # train_sample draws dx = jax.jacobian(geodesic, t); when the base draw x0
    # coincides with the target x1 the log_map's arccos/norm runs 0/0 under AD,
    # but the arc length is identically zero so the velocity limit is finite zero.
    geo = Spherical()
    x = jnp.array([0.6, 0.8, 0.0])
    jac = jax.jacobian(geo.geodesic)(0.3, x, x)
    assert jnp.all(jnp.isfinite(jac))
    assert jnp.allclose(jac, 0.0, atol=1e-4)


def test_geometry_tangent_dim_matches_chart_ambient():
    chart = SphericalChart(radius=2.0)
    geo = Spherical()
    p0 = jnp.array([0.3, 0.4])
    p1 = jnp.array([1.1, -0.2])
    x0, x1 = chart.forward(p0), chart.forward(p1)
    tangent = geo.log_map(x0, x1)
    assert tangent.shape[-1] == x0.shape[-1]


def test_isotropic_chart_roundtrip_sample_lies_on_sphere_of_correct_radius():
    chart = SphericalChart(radius=2.0)
    p = jnp.array([0.3, 1.1])
    x = chart.forward(p)
    assert jnp.allclose(jnp.linalg.norm(x), 2.0, atol=1e-4)
    assert jnp.allclose(chart.backward(x), p, atol=1e-4)


def test_near_antipodal_log_map_is_ill_conditioned():
    # just short of exactly antipodal, cos_angle isn't exactly -1 so dx isn't
    # exactly zero, but sinc(angle/pi) is still tiny: the tangent must stay
    # finite instead of blowing up near the exact-antipodal collapse.
    geo = Spherical()
    x0 = jnp.array([1.0, 0.0, 0.0])
    x1 = jnp.array([-1.0, 1e-6, 0.0])
    tangent = geo.log_map(x0, x1)
    assert jnp.all(jnp.isfinite(tangent))


def test_antipodal_fallback_axis_choice_high_dimension_e0_aligned():
    # x0 aligned with e0: |x0[0]|==r0, not < 0.9*r0, so the fallback must pick
    # axis 1 (e1) rather than degenerate onto x0's own direction
    geo = Spherical()
    x0 = jnp.array([2.0, 0.0, 0.0, 0.0, 0.0])
    x1 = -x0
    tangent = geo.log_map(x0, x1)
    assert jnp.all(jnp.isfinite(tangent))
    assert jnp.abs(jnp.dot(tangent, x0)) < 1e-4
    assert jnp.allclose(jnp.linalg.norm(tangent), 2.0 * jnp.pi, atol=1e-4)
    assert jnp.allclose(geo.geodesic(jnp.asarray(1.0), x0, x1), x1, atol=1e-4)


def test_antipodal_fallback_axis_choice_high_dimension_last_axis_aligned():
    # x0 aligned with a late axis (index 4): |x0[0]|==0 < 0.9*r0, so the
    # fallback must pick axis 0 (e0), which stays perpendicular
    geo = Spherical()
    x0 = jnp.array([0.0, 0.0, 0.0, 0.0, 2.0])
    x1 = -x0
    tangent = geo.log_map(x0, x1)
    assert jnp.all(jnp.isfinite(tangent))
    assert jnp.abs(jnp.dot(tangent, x0)) < 1e-4
    assert jnp.allclose(jnp.linalg.norm(tangent), 2.0 * jnp.pi, atol=1e-4)
    assert jnp.allclose(geo.geodesic(jnp.asarray(1.0), x0, x1), x1, atol=1e-4)


def test_antipodal_fallback_axis_choice_e1_aligned():
    # x0 aligned with e1: |x0[0]|==0 < 0.9*r0 picks axis 0, which is
    # perpendicular to x0 here (x0 has no e0 component)
    geo = Spherical()
    x0 = jnp.array([0.0, 1.5, 0.0])
    x1 = -x0
    tangent = geo.log_map(x0, x1)
    assert jnp.all(jnp.isfinite(tangent))
    assert jnp.abs(jnp.dot(tangent, x0)) < 1e-4
    assert jnp.allclose(jnp.linalg.norm(tangent), 1.5 * jnp.pi, atol=1e-4)


def test_antipodal_fallback_axis_selection_boundary_just_below_threshold():
    # |x0[0]| just under 0.9*r0: axis 0 branch selected; perp must still be
    # well-conditioned (not near-zero) even this close to the switch
    geo = Spherical()
    r0 = 1.0
    x0 = jnp.array([0.899 * r0, jnp.sqrt(1 - 0.899**2) * r0, 0.0])
    x1 = -x0
    tangent = geo.log_map(x0, x1)
    assert jnp.all(jnp.isfinite(tangent))
    assert jnp.abs(jnp.dot(tangent, x0)) < 1e-4
    assert jnp.linalg.norm(tangent) > 1e-3


def test_antipodal_fallback_axis_selection_boundary_just_above_threshold():
    # |x0[0]| just over 0.9*r0: axis 1 branch selected instead; must remain
    # well-conditioned across the discrete branch switch
    geo = Spherical()
    r0 = 1.0
    x0 = jnp.array([0.901 * r0, jnp.sqrt(1 - 0.901**2) * r0, 0.0])
    x1 = -x0
    tangent = geo.log_map(x0, x1)
    assert jnp.all(jnp.isfinite(tangent))
    assert jnp.abs(jnp.dot(tangent, x0)) < 1e-4
    assert jnp.linalg.norm(tangent) > 1e-3


def test_antipodal_fallback_axis_choice_independent_per_batch_row():
    # a single batched call mixing an e0-aligned and an e1-aligned antipodal
    # pair: each row's axis choice must be driven by its own x0, not the batch
    geo = Spherical()
    x0 = jnp.stack([jnp.array([1.0, 0.0, 0.0]), jnp.array([0.0, 1.0, 0.0])])
    x1 = -x0
    tangent = geo.log_map(x0, x1)
    assert tangent.shape == (2, 3)
    assert jnp.all(jnp.isfinite(tangent))
    assert jnp.abs(jnp.sum(tangent[0] * x0[0])) < 1e-4
    assert jnp.abs(jnp.sum(tangent[1] * x0[1])) < 1e-4
    assert jnp.allclose(jnp.linalg.norm(tangent, axis=-1), jnp.pi, atol=1e-4)


def test_batched_log_map_mixes_coincident_generic_and_antipodal_rows():
    # coincident, generic, and exact-antipodal rows in one batched call, each
    # must resolve independently to the right regime
    geo = Spherical()
    x0 = jnp.stack(
        [
            jnp.array([1.0, 0.0, 0.0]),
            jnp.array([1.0, 0.0, 0.0]),
            jnp.array([1.0, 0.0, 0.0]),
        ]
    )
    x1 = jnp.stack(
        [
            jnp.array([1.0, 0.0, 0.0]),
            jnp.array([0.6, 0.8, 0.0]),
            jnp.array([-1.0, 0.0, 0.0]),
        ]
    )
    tangent = geo.log_map(x0, x1)
    assert jnp.all(jnp.isfinite(tangent))
    assert jnp.allclose(tangent[0], 0.0, atol=1e-4)
    assert jnp.allclose(jnp.linalg.norm(tangent[2]), jnp.pi, atol=1e-4)


def test_geodesic_jacobian_wrt_t_finite_at_exact_antipode():
    # mirrors problems.py's jax.jacobian call at the cut locus itself
    geo = Spherical()
    x0 = jnp.array([1.0, 0.0, 0.0])
    x1 = jnp.array([-1.0, 0.0, 0.0])
    jac = jax.jacobian(geo.geodesic)(0.5, x0, x1)
    assert jnp.all(jnp.isfinite(jac))
    assert jac.shape == (3,)


def test_geodesic_jacobian_wrt_t_finite_near_antipode_high_dimension():
    geo = Spherical()
    x0 = jnp.array([1.0, 0.0, 0.0, 0.0])
    x1 = jnp.array([-1.0, 1e-6, 0.0, 0.0])
    jac = jax.jacobian(geo.geodesic)(0.5, x0, x1)
    assert jnp.all(jnp.isfinite(jac))


def test_log_map_tangent_when_x1_has_a_different_radius():
    # x1 only supplies a direction: the tangent must stay orthogonal to x0 and
    # keep magnitude r0*angle however far out x1 sits
    geo = Spherical()
    angle = 1.0
    direction = jnp.array([jnp.cos(angle), jnp.sin(angle), 0.0])
    x0 = jnp.array([1.0, 0.0, 0.0])
    for r1 in (0.5, 1.0, 3.0):
        tangent = geo.log_map(x0, r1 * direction)
        assert jnp.abs(jnp.dot(tangent, x0)) < 1e-6
        assert jnp.allclose(jnp.linalg.norm(tangent), angle, atol=1e-6)


def test_log_map_independent_of_x1_radius():
    geo = Spherical()
    x0 = jnp.array([2.0, 0.0, 0.0])
    x1 = jnp.array([0.3, 1.7, -0.9])
    assert jnp.allclose(geo.log_map(x0, x1), geo.log_map(x0, 4.0 * x1), atol=1e-6)


def test_geodesic_reaches_x1_direction_on_the_x0_sphere():
    # mismatched radii: the flow lives on the |x0| sphere, so it reaches x1's
    # direction at radius r0, not x1 itself
    geo = Spherical()
    x0 = jnp.array([2.0, 0.0, 0.0])
    x1 = 5.0 * jnp.array([0.0, 0.6, 0.8])
    out = geo.geodesic(jnp.asarray(1.0), x0, x1)
    assert jnp.allclose(jnp.linalg.norm(out), 2.0, atol=1e-6)
    assert jnp.allclose(out / 2.0, x1 / 5.0, atol=1e-6)
