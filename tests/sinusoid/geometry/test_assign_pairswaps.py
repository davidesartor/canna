"""The pairswap heuristic: it may not beat exhaustive search, but it must never regress its own start."""

import jax.numpy as jnp
import jax.random as jr
import pytest

from canna.sinusoid.geometries import Euclidean, Set, Spherical


def total_cost(geom, x0, x1, assignment):
    return float(jnp.sum(jnp.square(geom.log_map(x0, x1[assignment]))))


@pytest.mark.parametrize("n_points", [3, 4, 5, 6, 7, 8])
def test_assign_by_pairswaps_never_costs_more_than_its_starting_point(n_points):
    geom = Set(Euclidean(dim=2))
    for trial in range(5):
        x0 = jr.normal(jr.key(trial), (n_points, 2))
        x1 = jr.normal(jr.key(1000 + trial), (n_points, 2))
        start = geom.assign_by_rank(x0, x1)
        swapped = geom.assign_by_pairswaps(x0, x1)
        assert (
            total_cost(geom.local_geometry, x0, x1, swapped)
            <= total_cost(geom.local_geometry, x0, x1, start) + 1e-9
        )


@pytest.mark.parametrize("n_points", [3, 5, 7])
def test_assign_by_pairswaps_never_costs_more_than_its_starting_point_on_circle(
    n_points,
):
    geom = Set(Spherical(dim=2))
    for trial in range(5):
        angles0 = jr.uniform(jr.key(trial), (n_points,), maxval=2 * jnp.pi)
        angles1 = jr.uniform(jr.key(500 + trial), (n_points,), maxval=2 * jnp.pi)
        x0 = jnp.stack([jnp.cos(angles0), jnp.sin(angles0)], axis=-1)
        x1 = jnp.stack([jnp.cos(angles1), jnp.sin(angles1)], axis=-1)
        start = geom.assign_by_rank(x0, x1)
        swapped = geom.assign_by_pairswaps(x0, x1)
        assert (
            total_cost(geom.local_geometry, x0, x1, swapped)
            <= total_cost(geom.local_geometry, x0, x1, start) + 1e-9
        )


def test_assign_by_pairswaps_is_deterministic():
    geom = Set(Euclidean(dim=2))
    x0 = jr.normal(jr.key(0), (6, 2))
    x1 = jr.normal(jr.key(1), (6, 2))
    assert jnp.array_equal(
        geom.assign_by_pairswaps(x0, x1), geom.assign_by_pairswaps(x0, x1)
    )
