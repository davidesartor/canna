from abc import abstractmethod
from jaxtyping import Array, Key
import equinox as eqx

from .geometries import Geometry
from .charts import Chart


class Prior[Physical: Array](eqx.Module):
    """A distribution over one parameter block, carrying its own geometry and chart."""

    @abstractmethod
    def __call__(self, key: Key[Array, ""]) -> Physical: ...

    @property
    @abstractmethod
    def geometry(self) -> Geometry: ...

    @property
    @abstractmethod
    def chart(self) -> Chart: ...
