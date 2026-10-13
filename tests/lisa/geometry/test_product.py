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
