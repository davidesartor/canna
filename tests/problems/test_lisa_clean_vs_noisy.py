"""clean=True must mean exactly the noiseless signal, independent of key."""

import jax.numpy as jnp
import jax.random as jr

from canna.problems import LisaGB


def test_sample_observation_clean_true_matches_clean_signal():
    problem = LisaGB(n_sources=2)
    p = problem.sample_physical(jr.key(30))
    o_direct = problem.clean_signal(p)
    o_via_sample = problem.sample_observation(jr.key(31), p, clean=True)
    assert jnp.allclose(o_direct, o_via_sample, atol=1e-6, rtol=1e-6)


def test_sample_observation_clean_true_ignores_key():
    problem = LisaGB(n_sources=2)
    p = problem.sample_physical(jr.key(32))
    o1 = problem.sample_observation(jr.key(0), p, clean=True)
    o2 = problem.sample_observation(jr.key(1), p, clean=True)
    assert jnp.allclose(o1, o2, atol=1e-6, rtol=1e-6)


def test_sample_observation_noisy_differs_across_keys():
    # noise is nondegenerate (nonzero variance), so two independent draws
    # almost surely differ.
    problem = LisaGB(n_sources=2)
    p = problem.sample_physical(jr.key(33))
    o1 = problem.sample_observation(jr.key(0), p, clean=False)
    o2 = problem.sample_observation(jr.key(1), p, clean=False)
    assert not jnp.allclose(o1, o2, atol=1e-9, rtol=1e-9)
