"""Shape-signature contracts for NoisyPoint."""

import jax
import pytest

from canna.point import NoisyPoint


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
    assert problem.sample_observation(jax.random.key(1), p).shape == batch_shape + (
        dim,
    )
    assert problem.preprocess(p).shape == batch_shape + (dim,)


@pytest.mark.parametrize("batch_shape", [(), (4,), (2, 3)])
def test_log_likelihood_reduces_the_trailing_dim(batch_shape):
    dim = 3
    problem = NoisyPoint(dim=dim)
    p = jax.random.normal(jax.random.key(0), batch_shape + (dim,))
    o = problem.sample_observation(jax.random.key(1), p)
    assert problem.log_likelihood(p, o).shape == batch_shape


@pytest.mark.parametrize("dim", [1, 3])
def test_train_sample_field_shapes(dim):
    problem = NoisyPoint(dim=dim)
    s = problem.train_sample(jax.random.key(0))
    assert s.xt.shape == (dim,)
    assert s.dx.shape == (dim,)
    assert s.t.shape == ()
    assert s.y.shape == (dim,)


@pytest.mark.parametrize("batch", [1, 8])
def test_train_sample_vmaps_to_a_leading_batch_axis(batch):
    dim = 3
    problem = NoisyPoint(dim=dim)
    s = jax.vmap(problem.train_sample)(jax.random.split(jax.random.key(0), batch))
    assert s.xt.shape == (batch, dim)
    assert s.dx.shape == (batch, dim)
    assert s.t.shape == (batch,)
    assert s.y.shape == (batch, dim)
