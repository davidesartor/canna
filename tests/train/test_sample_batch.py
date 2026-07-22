"""sample_batch draws batch_size independent TrainSamples, stacked on a new leading axis."""

import jax
import jax.numpy as jnp
from flax import nnx

from canna.train import sample_batch
from conftest import X_DIM, Y_DIM


def test_batch_shapes_get_leading_axis(fake_problem):
    batch_size = 4
    batch = sample_batch(fake_problem, nnx.Rngs(0), batch_size)
    assert batch.xt.shape == (batch_size, X_DIM)
    assert batch.dx.shape == (batch_size, X_DIM)
    assert batch.t.shape == (batch_size,)
    assert batch.y.shape == (batch_size, Y_DIM)
    assert batch.x_target.shape == (batch_size, X_DIM)
    assert batch.y_target.shape == (batch_size, Y_DIM)


def test_batch_of_one_keeps_leading_axis(fake_problem):
    """batch_size=1 must not squeeze away the leading batch axis"""
    batch = sample_batch(fake_problem, nnx.Rngs(0), 1)
    assert batch.xt.shape == (1, X_DIM)


def test_batch_elements_are_independent_draws(fake_problem):
    """each of the batch_size samples should fork a distinct rng, not repeat the same draw"""
    batch = sample_batch(fake_problem, nnx.Rngs(0), 8)
    assert not jnp.allclose(batch.xt[0], batch.xt[1])


def test_same_seed_is_reproducible(fake_problem):
    """defect: identical fresh Rngs seed must reproduce the identical batch"""
    batch_a = sample_batch(fake_problem, nnx.Rngs(0), 4)
    batch_b = sample_batch(fake_problem, nnx.Rngs(0), 4)
    assert jnp.allclose(batch_a.xt, batch_b.xt)
    assert jnp.allclose(batch_a.y, batch_b.y)


def test_recompiles_cleanly_across_batch_sizes(real_state):
    """batch_size is a static_argname on sample_batch; switching it must retrace, not error
    or silently reuse a stale traced shape"""
    small = sample_batch(real_state.problem, real_state.rngs, 4)
    large = sample_batch(real_state.problem, real_state.rngs, 9)
    assert (
        small.xt.shape
        == (4,) + real_state.problem.train_sample(jax.random.key(0)).xt.shape
    )
    assert large.xt.shape[0] == 9
    assert small.xt.shape[1:] == large.xt.shape[1:]
