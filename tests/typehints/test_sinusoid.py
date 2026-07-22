"""Shape-signature contracts for NoisySinusoid."""

import jax
import jax.numpy as jnp
import pytest

from canna.problems.sinusoid import NoisySinusoid


def small(**kw):
    return NoisySinusoid(
        n_sources=kw.get("n_sources", 2),
        t_obs=256.0,
        sampling_step=1.0,
        wdm_freq_bands=8,
        patch_downsample=4,
    )


@pytest.mark.parametrize("S", [1, 3])
def test_sample_physical_and_point_shapes(S):
    problem = small(n_sources=S)
    assert problem.sample_physical(jax.random.key(0)).shape == (S, 3)
    assert problem.sample_point(jax.random.key(0)).shape == (S, 4)


@pytest.mark.parametrize("batch_shape", [(), (4,)])
def test_clean_signal_and_observation_shapes(batch_shape):
    problem = small()
    p = problem.sample_physical(jax.random.key(0))
    p = jnp.broadcast_to(p, batch_shape + p.shape)
    T = int(problem.t_obs / problem.sampling_step)
    assert problem.clean_signal(p).shape == batch_shape + (T, 2)
    obs = problem.sample_observation(jax.random.key(1), p, clean=False)
    assert obs.shape == batch_shape + (T, 2)
    assert problem.snr(p).shape == batch_shape


def test_preprocess_produces_time_freq_image():
    problem = small()
    o = problem.sample_observation(
        jax.random.key(1), problem.sample_physical(jax.random.key(0))
    )
    img = problem.preprocess(o)
    assert img.ndim == 3
    assert img.shape[-1] == 2
