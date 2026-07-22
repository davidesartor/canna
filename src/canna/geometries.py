from abc import abstractmethod
from jaxtyping import Array, Float
import jax.numpy as jnp
import equinox as eqx


class Geometry[Point: Array, Tangent: Array](eqx.Module):
    """Geometry of the manifold where the flow is defined."""

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

    def log_map(
        self, x0: Float[Array, "... D"], x1: Float[Array, "... D"]
    ) -> Float[Array, "... D"]:
        return x1 - x0

    def exp_map(
        self, x0: Float[Array, "... D"], dx: Float[Array, "... D"]
    ) -> Float[Array, "... D"]:
        return jnp.clip(x0 + dx, -1.0, 1.0)
