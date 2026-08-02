"""Set assignment: permutation validity, exact search, and how assign routes between them."""

import itertools

import jax.numpy as jnp
import jax.random as jr
import pytest

from canna.sinusoid.geometries import Euclidean, Set


def reference_best_permutation(geom, x0, x1):
    """Ground truth: minimise total squared geodesic cost over every permutation at once."""
    perms = jnp.array(list(itertools.permutations(range(x0.shape[-2]))))
    costs = jnp.sum(jnp.square(geom.log_map(x0, x1[perms])), axis=(-2, -1))
    return tuple(perms[jnp.argmin(costs)].tolist())


@pytest.mark.parametrize("n_points", [2, 3, 4, 5])
def test_assign_by_brute_force_matches_exhaustive_reference(n_points):
    geom = Set(Euclidean(dim=2))
    x0 = jr.normal(jr.key(n_points), (n_points, 2))
    x1 = jr.normal(jr.key(100 + n_points), (n_points, 2))
    best = reference_best_permutation(geom.local_geometry, x0, x1)
    assert tuple(geom.assign_by_brute_force(x0, x1).tolist()) == best


@pytest.mark.parametrize("n_points", [1, 2, 3, 5, 8])
def test_every_assignment_is_a_valid_permutation(n_points):
    geom = Set(Euclidean(dim=3), rank=lambda x: x[..., 0])
    x0 = jr.normal(jr.key(n_points), (n_points, 3))
    x1 = jr.normal(jr.key(200 + n_points), (n_points, 3))
    expected = list(range(n_points))
    for assignment in [
        geom.assign_by_rank(x0, x1),
        geom.assign_by_brute_force(x0, x1),
        geom.assign_by_pairswaps(x0, x1),
    ]:
        assert sorted(assignment.tolist()) == expected


@pytest.mark.parametrize("n_points", [2, 3, 4, 6, 7])
def test_every_assignment_is_the_identity_on_matching_sets(n_points):
    geom = Set(Euclidean(dim=2), rank=lambda x: x[..., 0])
    x0 = jr.normal(jr.key(n_points), (n_points, 2))
    for assignment in [
        geom.assign_by_rank(x0, x0),
        geom.assign_by_brute_force(x0, x0),
        geom.assign_by_pairswaps(x0, x0),
    ]:
        assert jnp.allclose(x0[assignment], x0)


def test_assign_by_rank_pairs_sets_by_sorted_order():
    geom = Set(Euclidean(dim=1), rank=lambda x: x[..., 0])
    x0 = jnp.array([[3.0], [1.0], [2.0]])
    x1 = jnp.array([[20.0], [30.0], [10.0]])
    # ranks pair as 1<->10, 2<->20, 3<->30
    assert jnp.allclose(
        x1[geom.assign_by_rank(x0, x1)], jnp.array([[30.0], [10.0], [20.0]])
    )


def test_assign_by_rank_without_a_rank_function_is_the_identity():
    geom = Set(Euclidean(dim=2))
    x0 = jr.normal(jr.key(0), (4, 2))
    x1 = jr.normal(jr.key(1), (4, 2))
    assert jnp.allclose(x1[geom.assign_by_rank(x0, x1)], x1)


def test_assign_routes_to_brute_force_at_the_limit():
    geom = Set(Euclidean(dim=2), brute_force_limit=4)
    x0 = jr.normal(jr.key(0), (4, 2))
    x1 = jr.normal(jr.key(1), (4, 2))
    assert jnp.allclose(geom.assign(x0, x1), x1[geom.assign_by_brute_force(x0, x1)])


def test_assign_routes_to_pairswaps_above_the_limit():
    geom = Set(Euclidean(dim=2), brute_force_limit=4)
    x0 = jr.normal(jr.key(0), (5, 2))
    x1 = jr.normal(jr.key(1), (5, 2))
    assert jnp.allclose(geom.assign(x0, x1), x1[geom.assign_by_pairswaps(x0, x1)])


@pytest.mark.parametrize("n_points", [1, 2, 3, 4, 5, 6])
def test_matched_cost_is_permutation_invariant_up_to_the_brute_force_limit(n_points):
    """Exact assignment sees only the unordered set, so reordering x1 cannot change the cost."""
    geom = Set(Euclidean(dim=2))
    x0 = jr.normal(jr.key(n_points), (n_points, 2))
    x1 = jr.normal(jr.key(300 + n_points), (n_points, 2))
    permuted = x1[jr.permutation(jr.key(n_points), n_points)]
    cost = jnp.sum(jnp.square(geom.log_map(x0, x1)))
    cost_permuted = jnp.sum(jnp.square(geom.log_map(x0, permuted)))
    assert jnp.allclose(cost, cost_permuted, atol=1e-6)
