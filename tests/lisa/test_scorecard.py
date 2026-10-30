"""The scorecard bookkeeping: source matching, widths in bins, and calibration flags."""

import jax.random as jr
import numpy as np
import pytest

from canna.lisa import config
from canna.lisa.injections import match_sources
from canna.lisa.problem import LisaGB, sample_f
from canna.lisa.scorecard import score
from canna.lisa.simulator import sample_physical


@pytest.fixture(scope="module")
def setup():
    problem = LisaGB(**config.load("narrow-B").problem)
    f = sample_f(problem, jr.key(0))
    truth = np.asarray(sample_physical(problem, jr.key(1), f))
    return problem, f, truth


def draws_around(truth, n, f0_sigma_bins, t_obs, shift_bins=0.0, seed=0):
    """Draws scattered in f0 around the truth, each with its sources in a random order."""
    rng = np.random.default_rng(seed)
    d = np.repeat(truth[None], n, axis=0)
    d[..., 0] += (shift_bins + f0_sigma_bins * rng.standard_normal(d.shape[:2])) / t_obs
    order = np.array([rng.permutation(truth.shape[0]) for _ in range(n)])
    return np.take_along_axis(d, order[:, :, None], axis=1)


def test_matching_undoes_any_relabelling(setup):
    problem, f, truth = setup
    shuffled = draws_around(truth, 64, 0.0, problem.t_obs)
    assert np.array_equal(match_sources(problem, shuffled, truth, f), np.repeat(truth[None], 64, 0))


def test_a_calibrated_gaussian_scores_as_calibrated(setup):
    problem, f, truth = setup
    rows = score(problem, draws_around(truth, 20_000, 0.2, problem.t_obs), truth, f)
    assert np.allclose(rows["width"], 0.2, rtol=0.05)
    assert np.all(np.abs(rows["offset"]) < 0.02)
    assert np.all(np.abs(rows["rank"] - 0.5) < 0.05)
    assert rows["in68"].all() and rows["in95"].all()
    assert np.all(rows["snr"] > 0) and np.allclose(rows["ideal"], np.sqrt(3) / (np.pi * rows["snr"]))


def test_a_shifted_posterior_shows_in_the_offset_rank_and_coverage(setup):
    problem, f, truth = setup
    # draws centred 3 widths low: the truth sits above almost all of them
    rows = score(problem, draws_around(truth, 20_000, 0.1, problem.t_obs, shift_bins=-0.3), truth, f)
    assert np.allclose(rows["offset"], -0.3, atol=0.01)
    assert np.all(rows["rank"] > 0.99)
    assert not rows["in68"].any() and not rows["in95"].any()


def test_a_whole_injection_scores_end_to_end(small_flow, capsys):
    """The same path the job takes per injection, on the tiny test network: finite rows of
    the right length, and a summary that prints for them."""
    from canna.lisa.scorecard import score_injection, summary

    problem, flow, _, _, f = small_flow
    latent = sample_physical(problem, jr.key(5), f)
    rows = score_injection(problem, flow, latent, f, jr.key(6), n_draws=8, ode_steps=2)
    assert all(len(v) == problem.n_sources for v in rows.values())
    assert all(np.all(np.isfinite(rows[k])) for k in ("snr", "width", "ideal", "offset", "rank"))
    summary(rows, np.ones(problem.n_sources, bool), "test")
    assert "SNR" in capsys.readouterr().out
