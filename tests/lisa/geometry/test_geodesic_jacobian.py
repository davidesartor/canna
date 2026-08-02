"""d/dt geodesic(t, x0, x1) is the flow-matching target, so it must stay finite even
when the base draw coincides with the target and the curved log_map runs 0/0 under AD.
"""

import jax
import jax.numpy as jnp
import pytest

from canna.lisa.geometries import Bounded, Euclidean, Product, Set, Spherical

GEOMETRIES = {
    "euclidean": (Euclidean(3), jnp.array([0.3, -1.2, 0.7])),
    "bounded": (Bounded(2), jnp.array([-0.4, 0.7])),
    "circle": (Spherical(2), jnp.array([0.6, 0.8])),
    "spherical": (Spherical(3), jnp.array([0.0, 0.0, 1.0])),
    "product": (
        Product(Bounded(1), Spherical(2), Spherical(3)),
        jnp.array([0.5, 0.6, 0.8, 0.0, 1.0, 0.0]),
    ),
    "set": (
        Set(Product(Bounded(1), Spherical(2))),
        jnp.array([[0.5, 0.6, 0.8], [-0.3, 0.0, 1.0]]),
    ),
}


@pytest.mark.parametrize("name", list(GEOMETRIES))
@pytest.mark.parametrize("t", [0.0, 0.5, 1.0])
def test_geodesic_jacobian_finite_at_coincident_endpoints(name, t):
    geometry, x = GEOMETRIES[name]
    dx = jax.jacobian(geometry.geodesic)(jnp.asarray(t), x, x)
    assert jnp.all(jnp.isfinite(dx))
    assert jnp.allclose(dx, 0.0, atol=1e-4)


@pytest.mark.parametrize("name", list(GEOMETRIES))
def test_geodesic_jacobian_finite_at_distinct_endpoints(name):
    geometry, x0 = GEOMETRIES[name]
    x1 = geometry.exp_map(x0, 0.3 * jnp.ones_like(x0))
    for t in [0.0, 0.5, 1.0]:
        dx = jax.jacobian(geometry.geodesic)(jnp.asarray(t), x0, x1)
        assert jnp.all(jnp.isfinite(dx))


@pytest.mark.parametrize("name", list(GEOMETRIES))
def test_geodesic_endpoints(name):
    geometry, x0 = GEOMETRIES[name]
    x1 = geometry.exp_map(x0, 0.3 * jnp.ones_like(x0))
    assert jnp.allclose(geometry.geodesic(jnp.asarray(0.0), x0, x1), x0, atol=1e-6)
    assert jnp.allclose(geometry.geodesic(jnp.asarray(1.0), x0, x1), x1, atol=1e-5)
