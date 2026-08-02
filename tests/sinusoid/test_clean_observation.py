"""y_target is the noiseless observation: clean_signal, never the noisy draw."""

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import pytest

from canna.sinusoid import NoisySinusoid


@pytest.fixture(scope="module")
def problem():
    return NoisySinusoid(n_sources=2)


@pytest.fixture
def key():
    return jr.key(0)


def test_clean_signal_ignores_noise_level_magnitude(key):
    p = NoisySinusoid(n_sources=2).sample_physical(key)
    huge = NoisySinusoid(n_sources=2, noise_level=1e12)
    zero = NoisySinusoid(n_sources=2, noise_level=0.0)
    assert jnp.allclose(huge.clean_signal(p), zero.clean_signal(p), atol=1e-6)
    assert jnp.allclose(
        huge.clean_signal(p), zero.sample_observation(key, p), atol=1e-6
    )


def test_clean_signal_preserves_dtype(problem, key):
    p = problem.sample_physical(key)
    assert problem.clean_signal(p).dtype == jnp.float64


def test_clean_signal_batches_over_leading_p_axis(problem, key):
    keys = jr.split(key, 4)
    p_batch = jnp.stack([problem.sample_physical(k) for k in keys])
    o = problem.clean_signal(p_batch)
    assert o.shape[0] == 4
    assert o.shape[1:] == problem.clean_signal(p_batch[0]).shape
    assert o.shape == problem.sample_observation(key, p_batch).shape


def test_clean_signal_matches_eager_under_jit(problem, key):
    p = problem.sample_physical(key)
    jitted = eqx.filter_jit(problem.clean_signal)
    assert jnp.allclose(jitted(p), problem.clean_signal(p), atol=1e-6)


# --- 4-way split provenance: y <- key_o (2nd); y_target takes no key ---


def test_y_matches_second_subkey_key_o(problem, key):
    key_p, key_o, *_ = jr.split(key, 4)
    p = problem.sample_physical(key_p)
    expected_y = problem.preprocess(problem.sample_observation(key_o, p))
    assert jnp.allclose(problem.train_sample(key).y, expected_y, atol=1e-4)


def test_y_target_matches_clean_signal_of_key_p(problem, key):
    key_p, *_ = jr.split(key, 4)
    p = problem.sample_physical(key_p)
    expected_y_target = problem.preprocess(problem.clean_signal(p))
    assert jnp.allclose(
        problem.train_sample(key).y_target, expected_y_target, atol=1e-4
    )
