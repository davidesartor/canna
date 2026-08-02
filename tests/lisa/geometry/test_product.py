import itertools

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from canna.lisa.geometries import (
    Euclidean,
    Geometry,
    Product as ProductGeometry,
    Set as SetGeometry,
    Spherical,
)


class _MarkedGeom(Geometry):
    """Toy geometry: log/exp scaled by a distinct marker per instance, still a true inverse pair."""

    dim: int = eqx.field(static=True, default=1)
    marker: float = eqx.field(static=True, default=1.0)

    def log_map(self, x0, x1):
        return (x1 - x0) * self.marker

    def exp_map(self, x0, dx):
        return x0 + dx / self.marker


# --- ProductGeometry ---


def test_product_geometry_dim_is_sum_of_blocks():
    g = ProductGeometry(Euclidean(dim=2), Euclidean(dim=3))
    assert g.dim == 5


def test_product_geometry_dim_empty_is_zero():
    g = ProductGeometry()
    assert g.dim == 0


def test_product_geometry_single_block_matches_raw_geometry():
    x0 = jnp.array([1.0, 2.0, 3.0])
    x1 = jnp.array([4.0, -1.0, 0.5])
    g = ProductGeometry(Euclidean(dim=3))
    e = Euclidean(dim=3)
    assert jnp.allclose(g.log_map(x0, x1), e.log_map(x0, x1))
    assert jnp.allclose(g.exp_map(x0, x1 - x0), e.exp_map(x0, x1 - x0))


def test_product_geometry_log_map_does_not_leak_across_blocks():
    # two blocks with distinct markers: a wrong split point would scale the
    # wrong slice by the wrong marker and change the numeric result.
    a = _MarkedGeom(dim=2, marker=1.0)
    b = _MarkedGeom(dim=2, marker=3.0)
    g = ProductGeometry(a, b)
    x0 = jnp.zeros(4)
    x1 = jnp.array([1.0, 2.0, 3.0, 4.0])
    expected = jnp.concatenate([a.log_map(x0[:2], x1[:2]), b.log_map(x0[2:], x1[2:])])
    assert jnp.allclose(g.log_map(x0, x1), expected)


def test_product_geometry_roundtrip_exp_of_log():
    a = _MarkedGeom(dim=2, marker=2.0)
    b = _MarkedGeom(dim=3, marker=0.5)
    g = ProductGeometry(a, b)
    x0 = jnp.array([0.0, 0.0, 1.0, 1.0, 1.0])
    x1 = jnp.array([1.0, -1.0, 2.0, 0.0, 5.0])
    dx = g.log_map(x0, x1)
    assert jnp.allclose(g.exp_map(x0, dx), x1)


def test_product_geometry_geodesic_endpoints():
    g = ProductGeometry(Euclidean(dim=2), Euclidean(dim=2))
    x0 = jnp.array([0.0, 1.0, 2.0, 3.0])
    x1 = jnp.array([4.0, 5.0, 6.0, 7.0])
    assert jnp.allclose(g.geodesic(jnp.array(0.0), x0, x1), x0)
    assert jnp.allclose(g.geodesic(jnp.array(1.0), x0, x1), x1)


def test_product_geometry_batch_leading_axes():
    g = ProductGeometry(Euclidean(dim=2), Euclidean(dim=3))
    x0 = jnp.zeros((4, 4, 5))
    x1 = jnp.ones((4, 4, 5))
    out = g.log_map(x0, x1)
    assert out.shape == (4, 4, 5)


def test_product_geometry_named_kwargs_order():
    # named blocks assumed appended after positional ones, in call order
    e2 = Euclidean(dim=2)
    e3 = Euclidean(dim=3)
    g = ProductGeometry(e2, phase=e3)
    assert g.local_geometries == (e2, e3)


# --- SetGeometry ---


def test_set_geometry_dim_matches_local_dim():
    # SetGeometry.dim reflects per-element dim (S not stored on the geometry)
    geo = SetGeometry(Euclidean(dim=3))
    assert geo.dim == 3


def test_set_geometry_log_map_is_elementwise_no_reassignment():
    geo = SetGeometry(Euclidean(dim=1))
    x0 = jnp.array([[0.0], [10.0], [20.0]])
    x1 = jnp.array([[1.0], [12.0], [23.0]])
    out = geo.log_map(x0, x1)
    assert jnp.allclose(out, x1 - x0)


def test_set_geometry_log_map_equivariant_under_joint_permutation():
    geo = SetGeometry(Euclidean(dim=1))
    x0 = jnp.array([[0.0], [10.0], [20.0]])
    x1 = jnp.array([[1.0], [12.0], [23.0]])
    perm = jnp.array([2, 0, 1])
    out = geo.log_map(x0, x1)
    out_perm = geo.log_map(x0[perm], x1[perm])
    assert jnp.allclose(out_perm, out[perm])


def test_set_geometry_singleton_matches_local_geometry():
    geo = SetGeometry(Euclidean(dim=2))
    x0 = jnp.array([[1.0, 2.0]])
    x1 = jnp.array([[3.0, -1.0]])
    e = Euclidean(dim=2)
    assert jnp.allclose(geo.log_map(x0, x1)[0], e.log_map(x0[0], x1[0]))


def test_set_geometry_batch_leading_axes():
    geo = SetGeometry(Euclidean(dim=2))
    x0 = jnp.zeros((5, 4, 2))
    x1 = jnp.ones((5, 4, 2))
    out = geo.log_map(x0, x1)
    assert out.shape == (5, 4, 2)


def test_set_geometry_assign_returns_permutation_of_x1_points():
    geo = SetGeometry(Euclidean(dim=1))
    x0 = jnp.array([[3.0], [1.0], [2.0]])
    x1 = jnp.array([[10.0], [30.0], [20.0]])
    out = geo.assign(x0, x1)
    assert jnp.allclose(jnp.sort(out[:, 0]), jnp.sort(x1[:, 0]))


def test_set_geometry_assign_identity_on_matching_sets():
    geo = SetGeometry(Euclidean(dim=1))
    x0 = jnp.array([[3.0], [1.0], [2.0]])
    out = geo.assign(x0, x0)
    assert jnp.allclose(out, x0)


def test_set_geometry_assign_recovers_known_permutation():
    geo = SetGeometry(Euclidean(dim=1), brute_force_limit=6)
    x0 = jnp.array([[3.0], [1.0], [2.0]])
    perm = jnp.array([2, 0, 1])
    x1 = x0[perm]
    out = geo.assign(x0, x1)
    assert jnp.allclose(out, x0)


def test_set_geometry_assign_by_rank_matches_sorted_order():
    geo = SetGeometry(Euclidean(dim=1), rank=lambda x: x[..., 0])
    x0 = jnp.array([[3.0], [1.0], [2.0]])
    x1 = jnp.array([[20.0], [10.0], [30.0]])
    idx = geo.assign_by_rank(x0, x1)
    order0 = jnp.argsort(x0[:, 0])
    order1 = jnp.argsort(x1[:, 0])
    expected = jnp.zeros_like(order0).at[order0].set(order1)
    assert jnp.array_equal(idx, expected)


def test_set_geometry_assign_by_rank_is_valid_permutation():
    geo = SetGeometry(Euclidean(dim=1), rank=lambda x: x[..., 0])
    x0 = jnp.array([[3.0], [1.0], [2.0], [5.0]])
    x1 = jnp.array([[20.0], [10.0], [30.0], [40.0]])
    idx = geo.assign_by_rank(x0, x1)
    assert jnp.array_equal(jnp.sort(idx), jnp.arange(4))


def test_set_geometry_assign_by_brute_force_minimizes_squared_cost():
    geo = SetGeometry(Euclidean(dim=1), brute_force_limit=6)
    x0 = jnp.array([[0.0], [5.0], [10.0]])
    x1 = jnp.array([[10.5], [0.2], [4.8]])
    idx = geo.assign_by_brute_force(x0, x1)

    def cost(perm):
        return sum((float(x1[p, 0]) - float(x0[k, 0])) ** 2 for k, p in enumerate(perm))

    best = min(itertools.permutations(range(3)), key=cost)
    got = tuple(int(i) for i in idx)
    assert cost(got) == pytest.approx(cost(best))


def test_set_geometry_assign_by_brute_force_identity_when_sets_match():
    geo = SetGeometry(Euclidean(dim=1), brute_force_limit=6)
    x0 = jnp.array([[0.0], [5.0], [10.0]])
    idx = geo.assign_by_brute_force(x0, x0)
    assert jnp.array_equal(jnp.sort(idx), jnp.arange(3))
    assert jnp.allclose(x0[idx], x0)


def test_set_geometry_assign_by_brute_force_is_valid_permutation_with_duplicates():
    geo = SetGeometry(Euclidean(dim=1), brute_force_limit=6)
    x0 = jnp.array([[1.0], [1.0], [2.0]])
    x1 = jnp.array([[1.0], [2.0], [1.0]])
    idx = geo.assign_by_brute_force(x0, x1)
    assert jnp.array_equal(jnp.sort(idx), jnp.arange(3))


def test_set_geometry_assign_by_pairswaps_is_valid_permutation():
    geo = SetGeometry(Euclidean(dim=1))
    x0 = jnp.array([[0.0], [5.0], [10.0]])
    x1 = jnp.array([[10.5], [0.2], [4.8]])
    idx = geo.assign_by_pairswaps(x0, x1, key=jr.key(0))
    assert jnp.array_equal(jnp.sort(idx), jnp.arange(3))


def test_set_geometry_assign_by_pairswaps_deterministic_given_key():
    geo = SetGeometry(Euclidean(dim=1))
    x0 = jnp.array([[0.0], [5.0], [10.0]])
    x1 = jnp.array([[10.5], [0.2], [4.8]])
    idx_a = geo.assign_by_pairswaps(x0, x1, key=jr.key(0))
    idx_b = geo.assign_by_pairswaps(x0, x1, key=jr.key(0))
    assert jnp.array_equal(idx_a, idx_b)


# --- assign_by_rank, rank=None ---


def test_set_geometry_assign_by_rank_none_is_identity_broadcast_over_batch():
    geo = SetGeometry(Euclidean(dim=1))
    x0 = jnp.zeros((2, 4, 1))
    x1 = jnp.ones((2, 4, 1))
    idx = geo.assign_by_rank(x0, x1)
    expected = jnp.broadcast_to(jnp.arange(4, dtype=jnp.int32), (2, 4))
    assert jnp.array_equal(idx, expected)


# --- brute_force_limit routing ---


def test_set_geometry_assign_routes_to_brute_force_at_limit():
    geo = SetGeometry(Euclidean(dim=1), brute_force_limit=3)
    x0 = jnp.array([[0.0], [5.0], [10.0]])
    x1 = jnp.array([[10.5], [0.2], [4.8]])
    expected = jnp.take_along_axis(
        x1, geo.assign_by_brute_force(x0, x1)[..., None], axis=-2
    )
    assert jnp.array_equal(geo.assign(x0, x1), expected)


def test_set_geometry_assign_routes_to_pairswaps_above_limit_with_default_key():
    # assign() calls assign_by_pairswaps without a key: must match the function's
    # own default key=jr.key(0), not a fresh/independent one.
    geo = SetGeometry(Euclidean(dim=1), brute_force_limit=2)
    x0 = jnp.array([[0.0], [5.0], [10.0]])
    x1 = jnp.array([[10.5], [0.2], [4.8]])
    expected = jnp.take_along_axis(
        x1, geo.assign_by_pairswaps(x0, x1, key=jr.key(0))[..., None], axis=-2
    )
    assert jnp.allclose(geo.assign(x0, x1), expected)


def test_set_geometry_assign_identity_when_sets_match_beyond_brute_force_limit():
    # default brute_force_limit=6, S=7 forces the pairswaps route with odd S
    geo = SetGeometry(Euclidean(dim=1))
    x0 = (jnp.arange(7.0) * 3.0 + 1.0)[:, None]
    out = geo.assign(x0, x0)
    assert jnp.allclose(out, x0)


# --- batched vs vmap ---


def test_set_geometry_assign_by_brute_force_batched_matches_vmap_per_batch():
    geo = SetGeometry(Euclidean(dim=1), brute_force_limit=6)
    x0 = jnp.array([[[0.0], [5.0], [10.0]], [[1.0], [2.0], [3.0]]])
    x1 = jnp.array([[[10.5], [0.2], [4.8]], [[2.9], [1.1], [2.1]]])
    batched = geo.assign_by_brute_force(x0, x1)
    looped = jax.vmap(geo.assign_by_brute_force)(x0, x1)
    assert jnp.array_equal(batched, looped)


def test_set_geometry_assign_by_pairswaps_batched_matches_vmap_per_batch():
    geo = SetGeometry(Euclidean(dim=1))
    x0 = jnp.array([[[0.0], [5.0], [10.0]], [[1.0], [2.0], [3.0]]])
    x1 = jnp.array([[[10.5], [0.2], [4.8]], [[2.9], [1.1], [2.1]]])
    key = jr.key(0)
    batched = geo.assign_by_pairswaps(x0, x1, key=key)
    looped = jax.vmap(lambda a, b: geo.assign_by_pairswaps(a, b, key=key))(x0, x1)
    assert jnp.array_equal(batched, looped)


# --- composites over curved leaf geometries ---


def _circle_point(angle):
    return jnp.array([jnp.cos(angle), jnp.sin(angle)])


def _sphere_point(v):
    return v / jnp.linalg.norm(v)


def test_product_geometry_dim_with_curved_blocks():
    g = ProductGeometry(Spherical(dim=2), Spherical(dim=2), Spherical(dim=3))
    assert g.dim == 7


def test_product_geometry_over_curved_blocks_geodesic_endpoints():
    g = ProductGeometry(Spherical(dim=2), Spherical(dim=3))
    x0 = jnp.concatenate(
        [_circle_point(0.3), _sphere_point(jnp.array([1.0, 0.0, 0.0]))]
    )
    x1 = jnp.concatenate(
        [_circle_point(2.1), _sphere_point(jnp.array([0.0, 1.0, 0.0]))]
    )
    assert jnp.allclose(g.geodesic(jnp.array(0.0), x0, x1), x0, atol=1e-4)
    assert jnp.allclose(g.geodesic(jnp.array(1.0), x0, x1), x1, atol=1e-4)


def test_set_geometry_over_circle_dim_ignores_set_size():
    # dim reflects one element's flat width, S is not folded in
    geo = SetGeometry(Spherical(dim=4))
    assert geo.dim == 4


def test_set_geometry_over_circle_assign_identity_on_matching_sets():
    geo = SetGeometry(Spherical(dim=2))
    angles = jnp.array([0.3, 1.7, 4.0])
    x0 = jnp.stack([_circle_point(a) for a in angles])
    out = geo.assign(x0, x0)
    assert jnp.allclose(out, x0, atol=1e-4)


def test_set_geometry_over_spherical_singleton_matches_local_geometry():
    sph = Spherical(dim=3)
    geo = SetGeometry(sph)
    x0 = _sphere_point(jnp.array([1.0, 0.0, 0.0]))[None, :]
    x1 = _sphere_point(jnp.array([0.0, 1.0, 0.0]))[None, :]
    out = geo.log_map(x0, x1)
    assert jnp.allclose(out[0], sph.log_map(x0[0], x1[0]), atol=1e-4)
