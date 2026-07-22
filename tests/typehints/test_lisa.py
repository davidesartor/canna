"""Shape-signature contracts for LisaGB's light methods (heavy signal/preprocess are in test_lisa_shapes)."""

import jax
import jax.numpy as jnp
import pytest

from canna.problems.lisa import LisaGB


@pytest.fixture(scope="module")
def problem():
    return LisaGB(n_sources=1)


@pytest.mark.parametrize("S", [1, 3])
def test_sample_physical_and_point_shapes(S):
    problem = LisaGB(n_sources=S)
    assert problem.sample_physical(jax.random.key(0)).shape == (S, 8)
    assert problem.sample_point(jax.random.key(0)).shape == (S, 11)


@pytest.mark.parametrize("batch_shape", [(), (4,), (2, 3)])
def test_noise_psd_appends_three_channels(problem, batch_shape):
    f = jax.random.uniform(jax.random.key(0), batch_shape, minval=1e-4, maxval=1e-2)
    assert problem.noise_psd(f).shape == batch_shape + (3,)


@pytest.mark.parametrize("batch_shape", [(), (5,)])
def test_snr_reduces_source_axis(problem, batch_shape):
    p = jax.random.uniform(
        jax.random.key(0), batch_shape + (1, 8), minval=1e-23, maxval=1e-22
    )
    assert problem.snr(p).shape == batch_shape
