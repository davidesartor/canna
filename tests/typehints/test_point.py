"""Shape-signature contracts for NoisyPoint and the Problem.train_sample contract."""

import jax
import jax.numpy as jnp
import pytest

from canna.problems.point import NoisyPoint


@pytest.mark.parametrize("dim", [1, 2, 5])
def test_sample_physical_and_point_return_dim(dim):
    problem = NoisyPoint(dim=dim)
    assert problem.sample_physical(jax.random.key(0)).shape == (dim,)
    assert problem.sample_point(jax.random.key(0)).shape == (dim,)


@pytest.mark.parametrize("batch_shape", [(), (4,), (2, 3)])
def test_sample_observation_preserves_batch_dim(batch_shape):
    dim = 3
    problem = NoisyPoint(dim=dim)
    p = jax.random.normal(jax.random.key(0), batch_shape + (dim,))
    noisy = problem.sample_observation(jax.random.key(1), p, clean=False)
    clean = problem.sample_observation(jax.random.key(1), p, clean=True)
    assert noisy.shape == batch_shape + (dim,)
    assert clean.shape == batch_shape + (dim,)
    assert problem.preprocess(p).shape == batch_shape + (dim,)


@pytest.mark.parametrize("dim", [1, 3])
def test_train_sample_field_shapes(dim):
    problem = NoisyPoint(dim=dim)
    s = problem.train_sample(jax.random.key(0))
    assert s.xt.shape == (dim,)
    assert s.dx.shape == (dim,)
    assert s.t.shape == ()
    assert s.y.shape == (dim,)
    assert s.x_target.shape == (dim,)
    assert s.y_target.shape == (dim,)
