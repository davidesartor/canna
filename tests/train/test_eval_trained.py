"""A flow trained to convergence must put its posterior draws around the truth."""

import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import pytest
from flax import nnx

from canna.eval import sample_posterior
from canna.networks.mlp import MLPFlow
from canna.problems import NoisyPoint
from canna.train import TrainState, sample_batch


def _make_state(seed: int = 0) -> TrainState:
    problem = NoisyPoint(noise_std=0.1)
    rngs = nnx.Rngs(seed)
    sample = problem.train_sample(rngs())
    flow = MLPFlow(
        x_shape=sample.xt.shape,
        y_shape=sample.y.shape,
        hidden_dim=64,
        num_blocks=2,
        rngs=rngs,
    )
    optimizer = nnx.Optimizer(flow, optax.adamw(1e-3, weight_decay=1e-5), wrt=nnx.Param)
    return TrainState(
        problem=problem,
        flow=flow,
        optimizer=optimizer,
        flow_metrics=nnx.metrics.Welford(),
        x_metrics=nnx.metrics.Welford(),
        y_metrics=nnx.metrics.Welford(),
        rngs=rngs,
    )


def _train(state: TrainState, steps: int) -> None:
    # flow head only; the reconstruction heads are irrelevant to the posterior draw
    weights = jnp.array([1.0, 0.0, 0.0])
    for _ in range(steps):
        batch = sample_batch(state.problem, state.rngs, 128)
        state.train_step(batch, weights)


@pytest.fixture(scope="module")
def trained() -> TrainState:
    state = _make_state(0)
    _train(state, 500)
    return state


def test_concentrates_near_truth(trained):
    problem = trained.problem
    key_p, key_o, key_u = jr.split(jr.key(5), 3)
    truth = problem.sample_physical(key_p)
    y = problem.preprocess(problem.sample_observation(key_o, truth))
    u0 = jax.vmap(problem.sample_point)(jr.split(key_u, 2000))
    post = sample_posterior(problem.geometry, trained.flow, u0, y, 8)
    physical = jax.vmap(problem.chart.backward)(post)
    assert jnp.allclose(jnp.mean(physical, axis=0), truth, atol=0.3)
    assert jnp.all(jnp.std(physical, axis=0) < 0.8)
