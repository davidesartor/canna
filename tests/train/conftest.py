"""Shared fixtures: a minimal deterministic Problem and a tiny real MLPFlow TrainState."""

import pytest
import jax
import jax.numpy as jnp
import optax
from flax import nnx

from canna.train import Problem, TrainSample, TrainState
from canna.networks.mlp import MLPFlow
from canna.charts import Affine
from canna.geometries import Euclidean
from canna.problems import NoisyPoint

X_DIM = 3
Y_DIM = 2


class _ConstantProblem(Problem):
    """Minimal Problem: draws an unbatched TrainSample of fixed shape from one key."""

    chart: Affine = Affine(shift=jnp.zeros(X_DIM), scale=jnp.ones(X_DIM))
    geometry: Euclidean = Euclidean(dim=X_DIM)

    def sample_physical(self, key):
        return jax.random.normal(key, (X_DIM,))

    def sample_point(self, key):
        return self.chart.forward(self.sample_physical(key))

    def sample_observation(self, key, p, clean: bool = False):
        return jax.random.normal(key, (Y_DIM,)) * (0.0 if clean else 1.0)

    def preprocess(self, o):
        return o

    def log_likelihood(self, p, o):
        return jnp.zeros(o.shape[:-1])

    def train_sample(self, key) -> TrainSample:
        k_xt, k_dx, k_t, k_y = jax.random.split(key, 4)
        xt = jax.random.normal(k_xt, (X_DIM,))
        dx = jax.random.normal(k_dx, (X_DIM,))
        t = jax.random.uniform(k_t, ())
        y = jax.random.normal(k_y, (Y_DIM,))
        return TrainSample(xt=xt, dx=dx, t=t, y=y, x_target=xt, y_target=y)


@pytest.fixture
def fake_problem():
    return _ConstantProblem()


@pytest.fixture
def make_state():
    def _make(seed: int = 0) -> TrainState:
        problem = _ConstantProblem()
        flow = MLPFlow(
            x_shape=(X_DIM,),
            y_shape=(Y_DIM,),
            hidden_dim=8,
            num_blocks=1,
            rngs=nnx.Rngs(seed),
        )
        optimizer = nnx.Optimizer(flow, optax.adam(1e-2), wrt=nnx.Param)
        return TrainState(
            problem=problem,
            flow=flow,
            optimizer=optimizer,
            flow_metrics=nnx.metrics.Welford(),
            x_metrics=nnx.metrics.Welford(),
            y_metrics=nnx.metrics.Welford(),
            rngs=nnx.Rngs(seed),
        )

    return _make


@pytest.fixture
def tiny_state(make_state):
    return make_state(0)


@pytest.fixture
def real_state():
    """A TrainState over a real NoisyPoint, for the parts that need genuine nnx members."""
    problem = NoisyPoint(noise_std=0.1)
    rngs = nnx.Rngs(0)
    sample = problem.train_sample(rngs())
    flow = MLPFlow(
        x_shape=sample.xt.shape,
        y_shape=sample.y.shape,
        hidden_dim=8,
        num_blocks=1,
        rngs=rngs,
    )
    optimizer = nnx.Optimizer(flow, optax.adam(1e-2), wrt=nnx.Param)
    return TrainState(
        problem=problem,
        flow=flow,
        optimizer=optimizer,
        flow_metrics=nnx.metrics.Welford(),
        x_metrics=nnx.metrics.Welford(),
        y_metrics=nnx.metrics.Welford(),
        rngs=rngs,
    )
