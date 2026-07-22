from abc import abstractmethod
from typing import NamedTuple
from jaxtyping import Array, Float, Key
import jax
import jax.random as jr
import equinox as eqx

from ..geometries import Geometry
from ..charts import Chart


class TrainSample[Point: Array, Tangent: Array, Conditioning: Array](NamedTuple):
    xt: Point
    dx: Tangent
    t: Float[Array, ""]
    y: Conditioning
    x_target: Point
    y_target: Conditioning


class Problem[
    Physical: Array,
    Observation: Array,
    Point: Array,
    Tangent: Array,
    Conditioning: Array,
](eqx.Module):
    """The forward model tying together priors, simulator, and parameter geometry."""

    chart: eqx.AbstractVar[Chart[Physical, Point]]
    geometry: eqx.AbstractVar[Geometry[Point, Tangent]]

    @abstractmethod
    def sample_physical(self, key: Key[Array, ""]) -> Physical: ...

    @abstractmethod
    def sample_point(self, key: Key[Array, ""]) -> Point: ...

    @abstractmethod
    def sample_observation(
        self, key: Key[Array, ""], p: Physical, clean: bool = False
    ) -> Observation: ...

    @abstractmethod
    def preprocess(self, o: Observation) -> Conditioning: ...

    @abstractmethod
    def log_likelihood(self, p: Physical, o: Observation) -> Float[Array, "..."]: ...

    def train_sample(
        self, key: Key[Array, ""]
    ) -> TrainSample[Point, Tangent, Conditioning]:
        key_p, key_o, key_o_target, key_x0, key_t = jr.split(key, 5)
        # sample and process physical quantities
        p = self.sample_physical(key_p)

        # noisy observation to condition on, clean one to reconstruct
        y = self.preprocess(self.sample_observation(key_o, p, clean=False))
        y_target = self.preprocess(self.sample_observation(key_o_target, p, clean=True))

        # sample and process flow quantities
        x0 = self.sample_point(key_x0)
        x1 = self.chart.forward(p)
        t = jr.uniform(key_t, ())
        xt = self.geometry.geodesic(t, x0, x1)
        dx = jax.jacobian(self.geometry.geodesic)(t, x0, x1)
        return TrainSample(xt=xt, dx=dx, t=t, y=y, x_target=x1, y_target=y_target)
