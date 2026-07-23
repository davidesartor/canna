"""sample_posterior's output shape, and a restored flow reproducing the saved one's outputs."""

import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import orbax.checkpoint as ocp
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
        hidden_dim=128,
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
def untrained() -> TrainState:
    return _make_state(0)


class TestSamplePosterior:
    def test_returns_point_shape_and_finite(self, untrained):
        problem = untrained.problem
        truth = problem.sample_physical(jr.key(2))
        y = problem.preprocess(problem.sample_observation(jr.key(1), truth))
        u0 = jax.vmap(problem.sample_point)(jr.split(jr.key(3), 256))
        post = sample_posterior(problem.geometry, untrained.flow, u0, y, 4)
        assert post.shape == (256, truth.shape[-1])
        assert jnp.all(jnp.isfinite(post))


class TestLoadFlow:
    def test_restored_flow_reproduces_outputs(self, tmp_path):
        state = _make_state(7)
        _train(state, 10)

        manager = ocp.CheckpointManager(tmp_path)
        state.save_to(manager, epoch=1, loss_hist=jnp.zeros((1, 1, 3)))
        manager.wait_until_finished()

        fresh = _make_state(999)
        manager2 = ocp.CheckpointManager(tmp_path)
        fresh.restore_from(manager2)

        batch = sample_batch(state.problem, nnx.Rngs(9), 8)
        ref = state.flow(batch.xt, batch.y, batch.t)[0]
        got = fresh.flow(batch.xt, batch.y, batch.t)[0]
        assert jnp.allclose(ref, got)
