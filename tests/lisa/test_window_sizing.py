"""clean_signal's F axis is exactly the sliding window, and preprocess keeps it: one row
of six channels per rfft bin."""

import warnings

import jax.random as jr
import pytest

from canna.lisa import config
from canna.lisa.problem import WINDOW_RESPONSES, LisaGB, window_freqs
from canna.lisa.simulator import clean_signal, preprocess, sample_physical
from ._helpers import window

# the bottom of the f0 prior is what limits how short a baseline can be: the response
# has to fit above DC (kmin >= 0), which a 1e-4 Hz source cannot do over a day
BAND = (3.0e-3, 12.0e-3)

# LisaGB-MMDiT-B.yaml's problem block, minus n_sources
SHIPPED = dict(
    n_sources=1,
    t_obs=63115200.0,
    f0_range=(1.0e-4, 12.0e-3),
    snr_range=(7.0, 1000.0),
)


def test_clean_signal_f_axis_is_the_window_width():
    problem = LisaGB(n_sources=1, t_obs=86400.0, f0_range=BAND)
    p = sample_physical(problem, jr.key(70), window(problem))
    o = clean_signal(problem, p, window(problem))
    assert o.shape[-2] == problem.window_bins
    assert problem.window_bins == WINDOW_RESPONSES * problem.response_points


def test_clean_signal_f_axis_does_not_move_with_the_baseline():
    # the window is a fixed bin count, so t_obs only changes which frequencies it covers
    day = LisaGB(n_sources=1, t_obs=86400.0, f0_range=BAND)
    week = LisaGB(n_sources=1, t_obs=7 * 86400.0, f0_range=BAND)
    p = sample_physical(day, jr.key(71), window(day))
    assert (
        clean_signal(day, p, window(day)).shape
        == clean_signal(week, p, window(week)).shape
    )
    assert float(window_freqs(day, window(day))[0]) != float(
        window_freqs(week, window(week))[0]
    )


def test_stream_shape_invariants_hold_across_a_range_of_observation_baselines():
    # preprocess consumes the whole window, so the stream is window_bins x 6 whether
    # t_obs is a day, the shipped 30-day baseline, or two years
    p_key = jr.key(69)
    for t_obs in (86400.0, 2592000.0, 2 * 365.25 * 24 * 60 * 60):
        problem = LisaGB(n_sources=1, t_obs=t_obs, f0_range=BAND)
        p = sample_physical(problem, p_key, window(problem))
        stream = preprocess(
            problem, clean_signal(problem, p, window(problem)), window(problem)
        )
        assert stream.shape == (problem.window_bins, 6)


def test_conditioning_shape_at_the_shipped_two_year_config():
    # regression pin for B.yaml: t_obs=2yr, response_points auto-sized to 1024, so
    # the window is 4096 bins, and LisaFlow's 16-bin tokens make 256 of them
    problem = LisaGB(**SHIPPED)
    p = sample_physical(problem, jr.key(60), window(problem))
    o = clean_signal(problem, p, window(problem))
    stream = preprocess(problem, o, window(problem))
    assert problem.response_points == 1024
    assert o.shape == (4096, 3)
    assert stream.shape == (4096, 6)


@pytest.mark.parametrize("band, response_points", [("narrow", 32), ("full", 1024)])
def test_each_band_autosizes_its_response_points(band, response_points):
    problem_config = config.load(f"{band}-B").problem
    assert "response_points" not in problem_config
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="response_points")
        problem = LisaGB(**problem_config)
    assert problem.response_points == response_points
    assert problem.f0_range == tuple(problem_config["f0_range"])
