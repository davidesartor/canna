from abc import abstractmethod
from jaxtyping import Array
import equinox as eqx


class Chart[Physical: Array, Point: Array](eqx.Module):
    """Invertible map between physical units and manifold coordinates."""

    @abstractmethod
    def forward(self, p: Physical) -> Point: ...

    @abstractmethod
    def backward(self, x: Point) -> Physical: ...
