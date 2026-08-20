"""Shape-signature contracts for NoisyPoint."""

import jax
import pytest

from canna.point import NoisyPoint


@pytest.mark.parametrize("dim", [1, 2, 5])
def test_sample_physical_and_point_return_dim(dim):
    problem = NoisyPoint(dim=dim)
    assert problem.sample_physical(jax.random.key(0)).shape == (dim,)
    assert problem.sample_flow(jax.random.key(0)).shape == (dim,)


@pytest.mark.parametrize("dim", [1, 3])
def test_sample_observation_and_preprocess_return_dim(dim):
    problem = NoisyPoint(dim=dim)
    p = jax.random.normal(jax.random.key(0), (dim,))
    assert problem.sample_observation(jax.random.key(1), p).shape == (dim,)
    assert problem.preprocess(p).shape == (dim,)


@pytest.mark.parametrize("dim", [1, 3])
def test_log_likelihood_is_a_scalar(dim):
    problem = NoisyPoint(dim=dim)
    p = jax.random.normal(jax.random.key(0), (dim,))
    o = problem.sample_observation(jax.random.key(1), p)
    assert problem.log_likelihood(p, o).shape == ()
