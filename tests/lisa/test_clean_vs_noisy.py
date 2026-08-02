"""clean_signal is the noiseless signal; sample_observation always adds noise."""

import jax.numpy as jnp
import jax.random as jr

from canna.lisa import LisaGB
from ._helpers import window



def test_clean_signal_ignores_key_by_taking_none():
    problem = LisaGB(n_sources=2)
    p = problem.sample_physical(jr.key(32), window(problem))
    assert jnp.allclose(
        problem.clean_signal(p, window(problem)), problem.clean_signal(p, window(problem)), atol=1e-6, rtol=1e-6
    )


def test_sample_observation_differs_from_clean_signal():
    problem = LisaGB(n_sources=2)
    p = problem.sample_physical(jr.key(30), window(problem))
    o_clean = problem.clean_signal(p, window(problem))
    o_noisy = problem.sample_observation(jr.key(31), p, window(problem))
    # strains are O(1e-17), so any absolute tolerance would swallow the whole draw
    assert not jnp.allclose(o_clean, o_noisy, atol=0.0, rtol=1e-6)


def test_sample_observation_differs_across_keys():
    # noise is nondegenerate (nonzero variance), so two independent draws
    # almost surely differ.
    problem = LisaGB(n_sources=2)
    p = problem.sample_physical(jr.key(33), window(problem))
    o1 = problem.sample_observation(jr.key(0), p, window(problem))
    o2 = problem.sample_observation(jr.key(1), p, window(problem))
    assert not jnp.allclose(o1, o2, atol=0.0, rtol=1e-6)
