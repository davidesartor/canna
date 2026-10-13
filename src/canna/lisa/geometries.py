import itertools
from abc import abstractmethod

from jaxtyping import Array, Float
import jax
import jax.numpy as jnp

import optax
import equinox as eqx


class Geometry[Point: Array, Tangent: Array](eqx.Module):
    """Geometry of the manifold where the flow is defined."""

    dim: eqx.AbstractVar[int]

    @abstractmethod
    def log_map(self, x0: Point, x1: Point) -> Tangent: ...

    @abstractmethod
    def exp_map(self, x0: Point, dx: Tangent) -> Point: ...

    def geodesic(self, t: Float[Array, ""], x0: Point, x1: Point) -> Point:
        # NOTE: dx is still in the tangent space
        dx: Tangent = t * self.log_map(x0, x1)  # type: ignore
        return self.exp_map(x0, dx)


class Euclidean(Geometry):
    """Flat space: geodesics are straight lines."""

    dim: int = eqx.field(static=True, default=1)

    def log_map(
        self, x0: Float[Array, "... D"], x1: Float[Array, "... D"]
    ) -> Float[Array, "... D"]:
        return x1 - x0

    def exp_map(
        self, x0: Float[Array, "... D"], dx: Float[Array, "... D"]
    ) -> Float[Array, "... D"]:
        return x0 + dx


class Bounded(Geometry):
    """Flat box [-1, 1]^D: geodesics are straight lines clipped to the box."""

    dim: int = eqx.field(static=True, default=1)

    def log_map(
        self, x0: Float[Array, "... D"], x1: Float[Array, "... D"]
    ) -> Float[Array, "... D"]:
        return x1 - x0

    def exp_map(
        self, x0: Float[Array, "... D"], dx: Float[Array, "... D"]
    ) -> Float[Array, "... D"]:
        return jnp.clip(x0 + dx, -1.0, 1.0)


class Spherical(Geometry):
    """Sphere of any radius: geodesics are great-circle arcs."""

    dim: int = eqx.field(static=True, default=2)

    def log_map(
        self, x0: Float[Array, "... D"], x1: Float[Array, "... D"]
    ) -> Float[Array, "... D"]:
        r0 = jnp.linalg.norm(x0, axis=-1, keepdims=True)
        r1 = jnp.linalg.norm(x1, axis=-1, keepdims=True)
        cos_angle = jnp.sum(x0 * x1, axis=-1, keepdims=True) / (r0 * r1)
        dx = x1 - cos_angle * r1 / r0 * x0
        norm = optax.safe_norm(dx, 0.0, axis=-1, keepdims=True)
        angle = jnp.arctan2(norm / r1, cos_angle)

        # antipodal cut locus: dx vanishes, and a pi rotation reaches x1 along
        # any direction, so substitute an arbitrary vector orthogonal to x0
        axis = jnp.where(jnp.abs(x0[..., :1]) < 0.9 * r0, 0, 1)
        e = (jnp.arange(x0.shape[-1]) == axis).astype(x0.dtype)
        perp = e - jnp.sum(e * x0, axis=-1, keepdims=True) / r0**2 * x0
        dx = jnp.where(norm > 0.0, dx, perp)
        tiny = jnp.finfo(dx.dtype).tiny
        return r0 * angle * dx / optax.safe_norm(dx, tiny, axis=-1, keepdims=True)

    def exp_map(
        self, x0: Float[Array, "... D"], dx: Float[Array, "... D"]
    ) -> Float[Array, "... D"]:
        r0 = jnp.linalg.norm(x0, axis=-1, keepdims=True)

        # drop any radial component: only the tangent part walks the great circle
        tangent = dx - jnp.sum(dx * x0, axis=-1, keepdims=True) / r0**2 * x0
        angle = optax.safe_norm(tangent, 0.0, axis=-1, keepdims=True) / r0
        return jnp.cos(angle) * x0 + jnp.sinc(angle / jnp.pi) * tangent


class Product(Geometry):
    """Product manifold: each block of coordinates moves on its own geometry."""

    local_geometries: tuple[Geometry, ...]

    def __init__(self, *geometries: Geometry, **named: Geometry):
        self.local_geometries = (*geometries, *named.values())

    @property
    def dim(self) -> int:
        return sum(g.dim for g in self.local_geometries)

    def log_map(
        self, x0: Float[Array, "... X"], x1: Float[Array, "... X"]
    ) -> Float[Array, "... X"]:
        dims = tuple(g.dim for g in self.local_geometries)
        edges = list(itertools.accumulate(dims))[:-1]
        blocks = zip(
            self.local_geometries,
            jnp.split(x0, edges, axis=-1),
            jnp.split(x1, edges, axis=-1),
        )
        return jnp.concat([g.log_map(a, b) for g, a, b in blocks], axis=-1)

    def exp_map(
        self, x0: Float[Array, "... X"], dx: Float[Array, "... X"]
    ) -> Float[Array, "... X"]:
        dims = tuple(g.dim for g in self.local_geometries)
        edges = list(itertools.accumulate(dims))[:-1]
        blocks = zip(
            self.local_geometries,
            jnp.split(x0, edges, axis=-1),
            jnp.split(dx, edges, axis=-1),
        )
        return jnp.concat([g.exp_map(a, d) for g, a, d in blocks], axis=-1)
