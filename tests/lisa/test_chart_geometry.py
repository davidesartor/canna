"""Whitening/geometry round trip and block layout."""

import itertools

import jax.numpy as jnp
import jax.random as jr

from canna.lisa import geometries
from canna.lisa import LisaGB, geodesic
from ._helpers import window


def test_chart_roundtrip_recovers_physical_parameters():
    problem = LisaGB(n_sources=2)
    p = problem.sample_physical(jr.key(1), window(problem))
    x = problem.physical_to_flow(p, window(problem))
    p_back = problem.flow_to_physical(x, window(problem))
    assert jnp.allclose(p_back, p, atol=1e-4, rtol=1e-4)


def test_physical_to_flow_maps_eight_to_eleven_per_source():
    problem = LisaGB(n_sources=2)
    p = problem.sample_physical(jr.key(2), window(problem))
    x = problem.physical_to_flow(p, window(problem))
    assert x.shape == (2, 11)


def test_physical_to_flow_equivariant_to_source_permutation():
    problem = LisaGB(n_sources=3)
    p = problem.sample_physical(jr.key(3), window(problem))
    perm = jnp.array([2, 0, 1])
    x = problem.physical_to_flow(p, window(problem))
    x_of_permuted = problem.physical_to_flow(p[perm], window(problem))
    assert jnp.allclose(x_of_permuted, x[perm], atol=1e-6, rtol=1e-6)


def test_chart_property_is_referentially_stable():
    problem = LisaGB(n_sources=2)
    p = problem.sample_physical(jr.key(4), window(problem))
    x1 = problem.physical_to_flow(p, window(problem))
    x2 = problem.physical_to_flow(p, window(problem))
    assert jnp.allclose(x1, x2)


def test_geometry_property_is_referentially_stable():
    problem = LisaGB(n_sources=2)
    x0 = problem.sample_flow(jr.key(5), window(problem))
    x1 = problem.sample_flow(jr.key(6), window(problem))
    v1 = problem.log_map(x0, x1)
    v2 = problem.log_map(x0, x1)
    assert jnp.allclose(v1, v2)


def test_geometry_log_exp_roundtrip_matches_up_to_source_permutation():
    # Set geometry is permutation-invariant, so exp_map(x0, log_map(x0, x1))
    # recovers *some* permutation of x1's rows, not necessarily x1's original order.
    problem = LisaGB(n_sources=3)
    x0 = problem.sample_flow(jr.key(14), window(problem))
    x1 = problem.sample_flow(jr.key(15), window(problem))
    v = problem.log_map(x0, x1)
    x1_recovered = problem.exp_map(x0, v)
    n = x1.shape[-2]
    assert any(
        jnp.allclose(x1_recovered, x1[jnp.array(perm)], atol=1e-4, rtol=1e-4)
        for perm in itertools.permutations(range(n))
    )


def test_geometry_log_map_zero_at_identity():
    problem = LisaGB(n_sources=2)
    x0 = problem.sample_flow(jr.key(9), window(problem))
    v = problem.log_map(x0, x0)
    assert jnp.allclose(v, jnp.zeros_like(v), atol=1e-6)


def test_problem_log_map_matches_its_geometry():
    problem = LisaGB(n_sources=2)
    x0 = problem.sample_flow(jr.key(11), window(problem))
    x1 = problem.sample_flow(jr.key(12), window(problem))
    lhs = problem.log_map(x0, x1)
    rhs = problem.geometry.log_map(x0, x1)
    assert jnp.allclose(lhs, rhs)


def test_geometry_blocks_are_in_the_jaxgb_order():
    problem = LisaGB(n_sources=2)
    blocks = problem.geometry.local_geometry.local_geometries
    assert [type(g).__name__ for g in blocks] == [
        "Bounded",
        "Euclidean",
        "Bounded",
        "Spherical",
        "Spherical",
        "Spherical",
    ]


def test_physical_blocks_sum_to_eight():
    problem = LisaGB(n_sources=2)
    assert problem.sample_physical(jr.key(10), window(problem)).shape == (2, 8)


def test_point_blocks_lift_the_spheres_to_vectors():
    problem = LisaGB(n_sources=2)
    dims = tuple(g.dim for g in problem.geometry.local_geometry.local_geometries)
    assert dims == (1, 1, 1, 3, 3, 2)


def test_sky_block_is_a_unit_vector():
    problem = LisaGB(n_sources=2)
    sky = problem.sample_flow(jr.key(0), window(problem))[:, 3:6]
    assert jnp.allclose(jnp.linalg.norm(sky, axis=-1), 1.0)


def test_orientation_block_is_a_unit_vector():
    problem = LisaGB(n_sources=2)
    orientation = problem.sample_flow(jr.key(0), window(problem))[:, 6:9]
    assert jnp.allclose(jnp.linalg.norm(orientation, axis=-1), 1.0)


def test_geometry_is_a_set_of_products():
    problem = LisaGB(n_sources=2)
    assert isinstance(problem.geometry, geometries.Set)
    assert isinstance(problem.geometry.local_geometry, geometries.Product)


def test_geodesic_starts_at_the_base_point():
    problem = LisaGB(n_sources=2)
    x0, x1 = problem.sample_flow(jr.key(3), window(problem)), problem.sample_flow(
        jr.key(4), window(problem)
    )
    assert jnp.allclose(geodesic(problem, jnp.array(0.0), x0, x1), x0)


def test_a_step_stays_on_the_spheres():
    problem = LisaGB(n_sources=2)
    x0, x1 = problem.sample_flow(jr.key(3), window(problem)), problem.sample_flow(
        jr.key(4), window(problem)
    )
    stepped = problem.exp_map(x0, problem.log_map(x0, x1))
    assert jnp.allclose(jnp.linalg.norm(stepped[:, 3:6], axis=-1), 1.0)
    assert jnp.allclose(jnp.linalg.norm(stepped[:, 6:9], axis=-1), 1.0)


def test_sample_flow_matches_physical_to_flow_of_sample_physical_same_key():
    # sample_flow's body is literally `physical_to_flow(sample_physical(key, f), f)`,
    # so given the same key it draws the same source set as sample_physical.
    problem = LisaGB(n_sources=2)
    key = jr.key(13)
    p = problem.sample_physical(key, window(problem))
    x = problem.sample_flow(key, window(problem))
    assert jnp.allclose(
        x, problem.physical_to_flow(p, window(problem)), atol=1e-4, rtol=1e-4
    )
