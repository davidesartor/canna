"""clean_signal's F axis is exactly the sliding window, and preprocess folds that window
into a wdm_times x wdm_freq_bands image."""

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
    wdm_freq_bands=256,
    wdm_times=32,
    patch_downsample=2,
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


def test_wdm_shape_invariants_hold_across_a_range_of_observation_baselines():
    # preprocess consumes the whole window, so the image axes are wdm_times and
    # wdm_freq_bands whether t_obs is a day, the shipped 30-day baseline, or two years
    p_key = jr.key(69)
    for t_obs in (86400.0, 2592000.0, 2 * 365.25 * 24 * 60 * 60):
        problem = LisaGB(n_sources=1, t_obs=t_obs, f0_range=BAND)
        p = problem.sample_physical(p_key, window(problem))
        img = problem.preprocess(
            problem.clean_signal(p, window(problem)), window(problem)
        )
        assert img.shape == (problem.wdm_times, problem.wdm_freq_bands, 3)
        assert img.shape[-3] % problem.patch_downsample == 0


def test_conditioning_image_shape_at_the_shipped_two_year_config():
    # regression pin for LisaGB-MMDiT-B.yaml: t_obs=2yr, wdm_times=32,
    # wdm_freq_bands=256, patch_downsample=2, response_points auto-sized
    problem = LisaGB(**SHIPPED)
    p = problem.sample_physical(jr.key(60), window(problem))
    o = problem.clean_signal(p, window(problem))
    img = problem.preprocess(o, window(problem))
    assert problem.response_points == 1024
    assert o.shape == (4096, 3)
    assert img.shape[-3] == 32
    assert img.shape[-2] == 256


def test_conditioning_frequency_axis_is_divisible_by_mmdit_b_patch_stages():
    # LisaGB-MMDiT-B.yaml sets patch_stages=1, so Patchify's single Fold2d halving needs
    # both image axes divisible by 2. The frequency axis is wdm_freq_bands (256) and the
    # time axis is wdm_times (32) -- both clear multiples of 2.
    problem = LisaGB(**SHIPPED)
    p = problem.sample_physical(jr.key(61), window(problem))
    o = problem.clean_signal(p, window(problem))
    img = problem.preprocess(o, window(problem))
    assert img.shape[-2] % 2 == 0
    assert img.shape[-3] % 2 == 0
