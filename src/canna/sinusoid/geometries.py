import itertools
from abc import abstractmethod
from typing import Callable, Optional
from jaxtyping import Array, Float, Int
import jax
import jax.numpy as jnp
import jax.random as jr
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


class Set(Geometry):
    """Interchangeable points on a shared geometry: the flow only sees the unordered set."""

    local_geometry: Geometry
    rank: Optional[Callable[[Float[Array, "... X"]], Float[Array, "..."]]] = eqx.field(
        static=True, default=None
    )
    brute_force_limit: int = eqx.field(static=True, default=6)

    @property
    def dim(self) -> int:
        return self.local_geometry.dim

    def log_map(
        self, x0: Float[Array, "... S X"], x1: Float[Array, "... S X"]
    ) -> Float[Array, "... S X"]:
        return self.local_geometry.log_map(x0, self.assign(x0, x1))

    def exp_map(
        self, x0: Float[Array, "... S X"], dx: Float[Array, "... S X"]
    ) -> Float[Array, "... S X"]:
        return self.local_geometry.exp_map(x0, dx)

    def assign(
        self, x0: Float[Array, "... S X"], x1: Float[Array, "... S X"]
    ) -> Float[Array, "... S X"]:
        exact = x1.shape[-2] <= self.brute_force_limit
        assign_by = self.assign_by_brute_force if exact else self.assign_by_pairswaps
        return jnp.take_along_axis(x1, assign_by(x0, x1)[..., None], axis=-2)

    def assign_by_rank(
        self, x0: Float[Array, "... S X"], x1: Float[Array, "... S X"]
    ) -> Int[Array, "... S"]:
        if self.rank is None:
            n_points = x0.shape[-2]
            return jnp.broadcast_to(
                jnp.arange(n_points, dtype=jnp.int32), x0.shape[:-1]
            )

        order0 = jnp.argsort(self.rank(x0), axis=-1)
        order1 = jnp.argsort(self.rank(x1), axis=-1)
        return jnp.take_along_axis(order1, jnp.argsort(order0, axis=-1), axis=-1)

    def assign_by_brute_force(
        self, x0: Float[Array, "... S X"], x1: Float[Array, "... S X"]
    ) -> Int[Array, "... S"]:
        n_points = x0.shape[-2]
        pairs = self.local_geometry.log_map(x1[..., :, None, :], x0[..., None, :, :])
        cost = jnp.sum(jnp.square(pairs), axis=-1)

        perms = jnp.array(list(itertools.permutations(range(n_points))))
        costs = jnp.sum(cost[..., perms, jnp.arange(n_points)], axis=-1)
        return perms[jnp.argmin(costs, axis=-1)]

    def assign_by_pairswaps(
        self,
        x0: Float[Array, "... S X"],
        x1: Float[Array, "... S X"],
        key: Array = jr.key(0),
    ) -> Int[Array, "... S"]:
        n_points = x0.shape[-2]
        n_paired = n_points - n_points % 2

        def cost(
            assigned: Int[Array, "... P"], slots: Int[Array, "P"]
        ) -> Float[Array, "... P"]:
            points = jnp.take_along_axis(x1, assigned[..., None], axis=-2)
            dx = self.local_geometry.log_map(points, x0[..., slots, :])
            return jnp.sum(jnp.square(dx), axis=-1)

        # randomly pair up the slots and swap any pair that lowers the total cost
        def sweep(i: int, a: Int[Array, "... S"]) -> Int[Array, "... S"]:
            shuffled = jr.permutation(jr.fold_in(key, i), n_points)
            p, q = shuffled[:n_paired].reshape(2, -1)
            ap, aq = a[..., p], a[..., q]
            gain = cost(ap, p) + cost(aq, q) - cost(aq, p) - cost(ap, q)
            swap = gain > 0.0
            return (
                a.at[..., p]
                .set(jnp.where(swap, aq, ap))
                .at[..., q]
                .set(jnp.where(swap, ap, aq))
            )

        start = self.assign_by_rank(x0, x1)
        return jax.lax.fori_loop(0, 8 * n_points.bit_length(), sweep, start)
