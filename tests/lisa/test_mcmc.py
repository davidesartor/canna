"""The joint MCMC's likelihood is the window's, exactly; its prior holds every source to its box."""

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from einops import rearrange

from canna.lisa.injections import eval_injections
from canna.lisa.mcmc import (
    ANGLES,
    log_likelihood,
    log_prior,
    make_window,
    to_physical,
    to_sampler,
)
from canna.lisa.problem import LisaGB
from canna.lisa.simulator import clean_signal, sample_physical, source_segments
from canna.lisa.simulator import log_likelihood as window_log_likelihood

NARROW = dict(n_sources=4, t_obs=63115200.0, response_points=32, f0_range=(1.0e-4, 1.2637070256260989e-3))


@pytest.fixture(scope="module")
def injections():
    return eval_injections(LisaGB(**NARROW), 0)


@pytest.fixture(scope="module")
def window(injections):
    return make_window(LisaGB(**NARROW), injections, 3)


def truth_theta(window):
    return np.stack([to_sampler(p, c) for p, c in zip(window.truth, window.centres)])


def test_joint_likelihood_matches_the_window(window):
    problem = LisaGB(**NARROW)
    t0 = truth_theta(window)
    flat = lambda t: jnp.asarray(rearrange(t, "s k -> (s k)"))
    rng = np.random.default_rng(0)
    for _ in range(5):
        # a step of a few bins in f0 and a broad one elsewhere, every source at once, inside the boxes
        t = t0 + rng.normal(size=t0.shape) * np.array([3e-5, 0.1, 0.1, 0.3, 0.1, 0.3, 0.1, 0.3])
        t = np.clip(t, window.lower + 1e-9, window.upper - 1e-9)
        joint = float(log_likelihood(problem, window, flat(t)) - log_likelihood(problem, window, flat(t0)))
        full = lambda t: window_log_likelihood(problem, to_physical(jnp.asarray(t)), window.obs, window.f)
        wide = float(full(t) - full(t0))
        assert np.isclose(joint, wide, rtol=1e-8, atol=1e-6), (joint, wide)


def test_prior_is_finite_at_the_truth_and_not_outside_a_box(window):
    t0 = truth_theta(window)
    assert np.isfinite(float(log_prior(window, jnp.asarray(t0.ravel()))))
    out = t0.copy()
    out[2, 4] = window.upper[2, 4] + 0.1
    assert float(log_prior(window, jnp.asarray(out.ravel()))) == -np.inf


def test_source_segments_rebuild_clean_signal(injections):
    problem = LisaGB(**NARROW)
    f = injections[1][0]
    p = sample_physical(problem, jr.key(3), f)
    idxs, segments = source_segments(problem, p, f)
    full = jnp.zeros((problem.window_bins, 3), segments.dtype).at[idxs].add(segments)
    assert jnp.allclose(full, clean_signal(problem, p, f))


def test_sampler_coordinates_round_trip(injections):
    problem = LisaGB(**NARROW)
    f = injections[1][0]
    p = np.asarray(sample_physical(problem, jr.key(4), f))[0]
    centres = np.zeros(8)
    centres[ANGLES] = p[ANGLES]
    back = np.asarray(to_physical(jnp.asarray(to_sampler(p, centres))))
    assert np.allclose(back[[0, 1, 2, 4, 6]], p[[0, 1, 2, 4, 6]], rtol=1e-12)
    assert np.allclose(np.angle(np.exp(1j * (back[ANGLES] - p[ANGLES]))), 0, atol=1e-12)
