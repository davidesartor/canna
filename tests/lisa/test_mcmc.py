"""The narrowband MCMC likelihood is the full-window likelihood of one source, exactly."""

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from canna.lisa.injections import eval_injections
from canna.lisa.mcmc import (
    ANGLES,
    log_likelihood,
    make_target,
    targets,
    to_physical,
    to_sampler,
)
from canna.lisa.plots import gb_picks
from canna.lisa.problem import LisaGB
from canna.lisa.simulator import (
    clean_signal,
    sample_observation,
    sample_physical,
    source_segments,
)
from canna.lisa.simulator import log_likelihood as window_log_likelihood

NARROW = dict(n_sources=4, t_obs=63115200.0, response_points=32, f0_range=(1.0e-4, 12.0e-3))


@pytest.fixture(scope="module")
def injections():
    return eval_injections(LisaGB(**NARROW), 0)


def test_narrowband_likelihood_matches_the_window(injections):
    problem = LisaGB(**NARROW)
    latents, windows = injections[0], injections[1]
    j, s = 3, 1
    src = make_target(problem, injections, j, s)
    obs = sample_observation(problem, injections[4][j], latents[j], windows[j])

    def full(p_s):
        p = jnp.asarray(latents[j]).at[s].set(p_s)
        return window_log_likelihood(problem, p, obs, windows[j])

    t0 = to_sampler(src.truth, src.centres)
    rng = np.random.default_rng(0)
    for _ in range(5):
        # a step of a few bins in f0 and a broad one elsewhere, inside the prior box
        t = t0 + rng.normal(size=8) * np.array([3e-5, 0.1, 0.1, 0.3, 0.1, 0.3, 0.1, 0.3])
        t = np.clip(t, src.lower + 1e-9, src.upper - 1e-9)
        narrow = float(
            log_likelihood(problem, src, jnp.asarray(t))
            - log_likelihood(problem, src, jnp.asarray(t0))
        )
        wide = float(full(to_physical(jnp.asarray(t))) - full(jnp.asarray(src.truth)))
        assert np.isclose(narrow, wide, rtol=1e-8, atol=1e-6), (narrow, wide)


def test_source_segments_rebuild_clean_signal(injections):
    problem = LisaGB(**NARROW)
    f = injections[1][0]
    p = sample_physical(problem, jr.key(3), f)
    idxs, segments = source_segments(problem, p, f)
    full = jnp.zeros((problem.window_bins, 3), segments.dtype).at[idxs].add(segments)
    assert jnp.allclose(full, clean_signal(problem, p, f))


def test_targets_cover_the_pages():
    snr = np.array([[5.0, 9.0, 30.0, 400.0], [7.0, 60.0, 2.0, 1000.0]])
    t = targets(snr)
    assert gb_picks(snr) == {"loudest": (1, 3), "median": (0, 2), "faintest": (1, 2)}
    assert set(t) == {(j, s) for j in range(2) for s in range(4)}
    assert t[0, 2] == ["gb-median", "q0.10"]
    assert t[1, 3] == ["gb-loudest", "q0.20"]


def test_sampler_coordinates_round_trip(injections):
    problem = LisaGB(**NARROW)
    f = injections[1][0]
    p = np.asarray(sample_physical(problem, jr.key(4), f))[0]
    centres = np.zeros(8)
    centres[ANGLES] = p[ANGLES]
    back = np.asarray(to_physical(jnp.asarray(to_sampler(p, centres))))
    assert np.allclose(back[[0, 1, 2, 4, 6]], p[[0, 1, 2, 4, 6]], rtol=1e-12)
    assert np.allclose(np.mod(back[ANGLES] - p[ANGLES], 2 * np.pi) % (2 * np.pi), 0, atol=1e-12)
