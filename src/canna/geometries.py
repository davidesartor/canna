from abc import abstractmethod
from jaxtyping import Array, Float
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
