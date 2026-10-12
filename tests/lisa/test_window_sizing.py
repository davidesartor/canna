"""clean_signal's F axis is exactly the sliding window, and preprocess keeps it: one row
of six channels per rfft bin."""

import jax.random as jr

from canna.lisa import LisaGB
from ._helpers import window

# the bottom of the f0 prior is what limits how short a baseline can be: the response
# has to fit above DC (kmin >= 0), which a 1e-4 Hz source cannot do over a day
BAND = (3.0e-3, 12.0e-3)

# LisaGB-MMDiT-B.yaml's problem block, minus n_sources
SHIPPED = dict(
    n_sources=1,
    t_obs=63115200.0,
    sampling_step=0.25,
    f0_range=(1.0e-4, 12.0e-3),
    snr_range=(7.0, 1000.0),
)


def test_clean_signal_f_axis_is_the_window_width():
    problem = LisaGB(n_sources=1, t_obs=86400.0, f0_range=BAND)
    p = problem.sample_physical(jr.key(70), window(problem))
    o = problem.clean_signal(p, window(problem))
    assert o.shape[-2] == problem.window_bins
    assert problem.window_bins == problem.wdm_freq_bands * (problem.wdm_times // 2)


def test_clean_signal_f_axis_does_not_move_with_the_baseline():
    # the window is a fixed bin count, so t_obs only changes which frequencies it covers
    day = LisaGB(n_sources=1, t_obs=86400.0, f0_range=BAND)
    week = LisaGB(n_sources=1, t_obs=7 * 86400.0, f0_range=BAND)
    p = day.sample_physical(jr.key(71), window(day))
    assert (
        day.clean_signal(p, window(day)).shape
        == week.clean_signal(p, window(week)).shape
    )
    assert float(day.window_freqs(window(day))[0]) != float(
        week.window_freqs(window(week))[0]
    )


def test_stream_shape_invariants_hold_across_a_range_of_observation_baselines():
    # preprocess consumes the whole window, so the stream is window_bins x 6 whether
    # t_obs is a day, the shipped 30-day baseline, or two years
    p_key = jr.key(69)
    for t_obs in (86400.0, 2592000.0, 2 * 365.25 * 24 * 60 * 60):
        problem = LisaGB(n_sources=1, t_obs=t_obs, f0_range=BAND)
        p = problem.sample_physical(p_key, window(problem))
        stream = problem.preprocess(
            problem.clean_signal(p, window(problem)), window(problem)
        )
        assert stream.shape == (problem.window_bins, 6)


def test_conditioning_shape_at_the_shipped_two_year_config():
    # regression pin for B-fd.yaml: t_obs=2yr, response_points auto-sized to 1024, so
    # the window is 4096 bins, and LisaFlow's 2-bin tokens make 2048 of them
    problem = LisaGB(**SHIPPED)
    p = problem.sample_physical(jr.key(60), window(problem))
    o = problem.clean_signal(p, window(problem))
    stream = problem.preprocess(o, window(problem))
    assert problem.response_points == 1024
    assert o.shape == (4096, 3)
    assert stream.shape == (4096, 6)
