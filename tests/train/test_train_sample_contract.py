"""TrainSample is the wire format between Problem and the training loop."""

import jax.numpy as jnp
import jax.random as jr

from canna.charts import Periodic
from canna.geometries import Toroidal
from canna.problems.base import Problem
from canna.train import TrainSample


def test_field_order_is_stable():
    """downstream code may positionally construct/unpack TrainSample; reordering breaks it silently"""
    assert TrainSample._fields == ("xt", "dx", "t", "y", "x_target", "y_target")


def test_dx_and_xt_share_trailing_dim():
    """dx is the velocity target at xt, so it must live in the same D as xt, not just any shape"""
    sample = TrainSample(
        xt=jnp.zeros((3,)),
        dx=jnp.zeros((3,)),
        t=jnp.array(0.0),
        y=jnp.zeros((2,)),
        x_target=jnp.zeros((3,)),
        y_target=jnp.zeros((2,)),
    )
    assert sample.dx.shape[-1] == sample.xt.shape[-1]
    assert sample.x_target.shape[-1] == sample.xt.shape[-1]


class _CoincidentToroidal(Problem):
    """Curved-geometry Problem whose base draw always lands on the target angle."""

    chart: Periodic = Periodic(period=jnp.array([2 * jnp.pi]))
    geometry: Toroidal = Toroidal()
    angle: float = 0.7

    def sample_physical(self, key):
        return jnp.array([self.angle])

    def sample_point(self, key):
        return self.chart.forward(self.sample_physical(key))

    def sample_observation(self, key, p, clean=False):
        return p

    def preprocess(self, o):
        return self.chart.forward(o)

    def log_likelihood(self, p, o):
        return jnp.zeros(o.shape[:-1])


def test_train_sample_dx_finite_when_base_draw_lands_on_target():
    """train_sample draws dx = jacobian(geodesic, t); if the base point coincides
    with the target the curved log_map runs 0/0 under AD, yet dx must stay finite."""
    problem = _CoincidentToroidal()
    for i in range(8):
        sample = problem.train_sample(jr.key(i))
        assert jnp.all(jnp.isfinite(sample.dx))
        assert jnp.allclose(sample.dx, 0.0, atol=1e-4)
