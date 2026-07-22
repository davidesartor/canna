"""Chart/geometry round trip and consistency with the prior."""

import itertools

import jax.numpy as jnp
import jax.random as jr

from canna import geometries
from canna.problems import LisaGB


def test_chart_roundtrip_recovers_physical_parameters():
    problem = LisaGB(n_sources=2)
    p = problem.sample_physical(jr.key(1))
    x = problem.chart.forward(p)
    p_back = problem.chart.backward(x)
    assert jnp.allclose(p_back, p, atol=1e-4, rtol=1e-4)


def test_chart_forward_maps_eight_to_eleven_per_source():
    problem = LisaGB(n_sources=2)
    p = problem.sample_physical(jr.key(2))
    x = problem.chart.forward(p)
    assert x.shape == (2, 11)


def test_chart_forward_equivariant_to_source_permutation():
    problem = LisaGB(n_sources=3)
    p = problem.sample_physical(jr.key(3))
    perm = jnp.array([2, 0, 1])
    x = problem.chart.forward(p)
    x_of_permuted = problem.chart.forward(p[perm])
    assert jnp.allclose(x_of_permuted, x[perm], atol=1e-6, rtol=1e-6)


def test_chart_property_is_referentially_stable():
    problem = LisaGB(n_sources=2)
    p = problem.sample_physical(jr.key(4))
    x1 = problem.chart.forward(p)
    x2 = problem.chart.forward(p)
    assert jnp.allclose(x1, x2)


def test_geometry_property_is_referentially_stable():
    problem = LisaGB(n_sources=2)
    x0 = problem.sample_point(jr.key(5))
    x1 = problem.sample_point(jr.key(6))
    v1 = problem.geometry.log_map(x0, x1)
    v2 = problem.geometry.log_map(x0, x1)
    assert jnp.allclose(v1, v2)


def test_geometry_log_exp_roundtrip_matches_up_to_source_permutation():
    # Set geometry is permutation-invariant, so exp_map(x0, log_map(x0, x1))
    # recovers *some* permutation of x1's rows, not necessarily x1's original order.
    problem = LisaGB(n_sources=3)
    x0 = problem.sample_point(jr.key(14))
    x1 = problem.sample_point(jr.key(15))
    v = problem.geometry.log_map(x0, x1)
    x1_recovered = problem.geometry.exp_map(x0, v)
    n = x1.shape[-2]
    assert any(
        jnp.allclose(x1_recovered, x1[jnp.array(perm)], atol=1e-4, rtol=1e-4)
        for perm in itertools.permutations(range(n))
    )


def test_geometry_log_map_zero_at_identity():
    problem = LisaGB(n_sources=2)
    x0 = problem.sample_point(jr.key(9))
    v = problem.geometry.log_map(x0, x0)
    assert jnp.allclose(v, jnp.zeros_like(v), atol=1e-6)


def test_problem_chart_matches_prior_chart():
    problem = LisaGB(n_sources=2)
    p = problem.sample_physical(jr.key(10))
    assert jnp.allclose(problem.chart.forward(p), problem.prior.chart.forward(p))


def test_problem_geometry_matches_prior_geometry():
    problem = LisaGB(n_sources=2)
    x0 = problem.sample_point(jr.key(11))
    x1 = problem.sample_point(jr.key(12))
    lhs = problem.geometry.log_map(x0, x1)
    rhs = problem.prior.geometry.log_map(x0, x1)
    assert jnp.allclose(lhs, rhs)


def test_prior_blocks_are_in_the_jaxgb_order():
    problem = LisaGB(n_sources=2)
    assert [type(p).__name__ for p in problem.prior.local_prior.local_priors] == [
        "LogUniform",
        "LogUniform",
        "LogUniform",
        "Isotropic",
        "Isotropic",
        "PeriodicUniform",
    ]


def test_physical_blocks_sum_to_eight():
    problem = LisaGB(n_sources=2)
    dims = tuple(c.physical_dim for c in problem.prior.chart.local_charts)
    assert dims == (1, 1, 1, 2, 2, 1)


def test_point_blocks_lift_the_spheres_to_vectors():
    problem = LisaGB(n_sources=2)
    dims = tuple(c.flow_dim for c in problem.prior.chart.local_charts)
    assert dims == (1, 1, 1, 3, 3, 2)


def test_sky_block_is_a_unit_vector():
    problem = LisaGB(n_sources=2)
    sky = problem.sample_point(jr.key(0))[:, 3:6]
    assert jnp.allclose(jnp.linalg.norm(sky, axis=-1), 1.0)


def test_orientation_block_is_a_unit_vector():
    problem = LisaGB(n_sources=2)
    orientation = problem.sample_point(jr.key(0))[:, 6:9]
    assert jnp.allclose(jnp.linalg.norm(orientation, axis=-1), 1.0)


def test_geometry_is_a_set_of_products():
    problem = LisaGB(n_sources=2)
    assert isinstance(problem.geometry, geometries.Set)
    assert isinstance(problem.geometry.local_geometry, geometries.Product)


def test_geodesic_starts_at_the_base_point():
    problem = LisaGB(n_sources=2)
    x0, x1 = problem.sample_point(jr.key(3)), problem.sample_point(jr.key(4))
    assert jnp.allclose(problem.geometry.geodesic(jnp.array(0.0), x0, x1), x0)


def test_a_step_stays_on_the_spheres():
    problem = LisaGB(n_sources=2)
    x0, x1 = problem.sample_point(jr.key(3)), problem.sample_point(jr.key(4))
    stepped = problem.geometry.exp_map(x0, problem.geometry.log_map(x0, x1))
    assert jnp.allclose(jnp.linalg.norm(stepped[:, 3:6], axis=-1), 1.0)
    assert jnp.allclose(jnp.linalg.norm(stepped[:, 6:9], axis=-1), 1.0)


def test_sample_point_matches_chart_forward_of_sample_physical_same_key():
    # sample_point's body is literally `self.chart.forward(self.sample_physical(key))`,
    # so given the same key it draws the same source set as sample_physical, bridged
    # by the chart.
    problem = LisaGB(n_sources=2)
    key = jr.key(13)
    p = problem.sample_physical(key)
    x = problem.sample_point(key)
    assert jnp.allclose(x, problem.chart.forward(p), atol=1e-4, rtol=1e-4)
