from abc import abstractmethod
from typing import NamedTuple
from jaxtyping import Array, Float, Key, Scalar
import jax
import jax.random as jr
import equinox as eqx

from .geometries import Geometry
from .charts import Chart


class TrainSample[Point: Array, Tangent: Array, Conditioning: Array](NamedTuple):
    xt: Point
    dx: Tangent
    t: Scalar
    y: Conditioning


class Problem[
    Physical: Array,
    Observation: Array,
    Point: Array,
    Tangent: Array,
    Conditioning: Array,
](eqx.Module):
    """The forward model tying together priors, simulator, and parameter geometry."""

    @property
    @abstractmethod
    def chart(self) -> Chart[Physical, Point]: ...

    @property
    @abstractmethod
    def geometry(self) -> Geometry[Point, Tangent]: ...

    @abstractmethod
    def sample_physical(self, key: Key) -> Physical: ...

    @abstractmethod
    def sample_point(self, key: Key) -> Point: ...

    @abstractmethod
    def sample_observation(self, key: Key, p: Physical) -> Observation: ...

    @abstractmethod
    def preprocess(self, o: Observation) -> Conditioning: ...

    @abstractmethod
    def log_likelihood(self, p: Physical, o: Observation) -> Float[Array, "..."]: ...

    def train_sample(self, key: Key) -> TrainSample[Point, Tangent, Conditioning]:
        key_p, key_o, key_x0, key_t = jr.split(key, 4)
        # sample and process physical quantities
        p = self.sample_physical(key_p)
        o = self.sample_observation(key_o, p)
        y = self.preprocess(o)

        # sample and process flow quantities
        x0 = self.sample_point(key_x0)
        x1 = self.chart.forward(p)
        t = jr.uniform(key_t, ())
        xt = self.geometry.geodesic(t, x0, x1)
        dx = jax.jacobian(self.geometry.geodesic)(t, x0, x1)
        return TrainSample(xt=xt, dx=dx, t=t, y=y)
