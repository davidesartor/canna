"""The conditioning band covers the whole f0 prior, and response_points covers a source."""

import jax
import jax.numpy as jnp
import pytest

from canna.problems import LisaGB
from canna.problems.lisa import fdot_from_chirp_mass

jax.config.update("jax_enable_x64", True)

MONTH = 2592000.0
V_OVER_C = 29785.0 / 299792458.0
CHIRP_MASS = 0.3  # a typical double white dwarf [Msun]


def test_top_of_the_f0_prior_lands_inside_the_conditioning_image():
    problem = LisaGB(n_sources=1, t_obs=MONTH)
    f0_hi = problem.f0_range[1]
    signal = problem.clean_signal(
        jnp.array([[f0_hi, CHIRP_MASS, 1e-22, 1.0, 0.3, 2.0, 0.5, 0.0]])
    )
    n_freq = signal.shape[-2]
    nyquist = n_freq / (2.0 * problem.t_obs)
    assert f0_hi < nyquist

    # the loudest pixel of the image must sit at the bin the source was injected at
    image = problem.preprocess(signal)
    nf = image.shape[-2]
    expected_bin = f0_hi / (nyquist / nf)
    assert abs(int(jnp.argmax(jnp.abs(image[..., 0]).max(axis=0))) - expected_bin) < 2


def test_no_signal_energy_lives_above_the_top_of_the_prior():
    problem = LisaGB(n_sources=1, t_obs=MONTH)
    f0_hi = problem.f0_range[1]
    signal = problem.clean_signal(
        jnp.array([[f0_hi, CHIRP_MASS, 1e-22, 1.0, 0.3, 2.0, 0.5, 0.0]])
    )
    n_freq = signal.shape[-2]
    power = jnp.abs(signal[:, 0]) ** 2
    band_end = int(f0_hi * problem.t_obs) + problem.response_points
    assert (
        float(power[band_end : n_freq // 2].sum() / power[: n_freq // 2].sum()) < 1e-12
    )


def test_the_band_the_prior_cannot_reach_is_a_small_fraction_of_the_image():
    # the (n // k + 1) * k quantisation overshoots the prior top; keep the waste bounded
    # at the 30-day mock config (wdm_freq_bands=4096), where the overshoot is small
    problem = LisaGB(n_sources=1, t_obs=MONTH, wdm_freq_bands=4096)
    signal = problem.clean_signal(
        jnp.array([[problem.f0_range[1], CHIRP_MASS, 1e-22, 1.0, 0.3, 2.0, 0.5, 0.0]])
    )
    n_freq = signal.shape[-2]
    nyquist = n_freq / (2.0 * problem.t_obs)
    assert problem.f0_range[1] / nyquist > 0.8


def test_a_source_at_either_band_edge_keeps_its_response_window_inside_the_band():
    # f0_range_bins pads fmin/fmax by response_points/2, so a source at either edge of
    # the f0 prior lands its whole response_points-wide window inside the kept band
    # instead of having power silently clipped by the out-of-bounds scatter add.
    problem = LisaGB(n_sources=1, t_obs=MONTH)
    fmin, fmax = problem.f0_range_bins
    band_bins = fmax - fmin + 1
    for edge in (problem.f0_range[0], problem.f0_range[1]):
        kmin_edge = int(problem.response.get_kmin(jnp.array([edge]))[0])
        start = kmin_edge - fmin
        assert start >= 0
        assert start + problem.response_points <= band_bins


def test_response_points_covers_the_annual_doppler_and_the_fdot_drift():
    # the chirp only drifts upward, so half the window has to hold doppler + drift
    problem = LisaGB(n_sources=1, t_obs=2 * 31557600.0)
    f0_hi, fdot_hi = problem.f0_range[1], problem.fdot_range[1]
    bin_width = 1.0 / problem.t_obs
    doppler_bins = f0_hi * V_OVER_C / bin_width
    drift_bins = fdot_hi * problem.t_obs / bin_width
    assert problem.response_points / 2.0 > doppler_bins + drift_bins


def test_a_too_narrow_response_window_for_the_chirp_is_rejected_at_init():
    # the loudest chirp the prior allows sits at the top of the f0 band with the
    # heaviest pair; at a two-year baseline it needs far more than 256 bins
    with pytest.raises(AssertionError, match="response_points"):
        LisaGB(n_sources=1, t_obs=2 * 31557600.0, response_points=256)


def test_the_local_response_segment_decays_to_its_edges():
    # if response_points were too narrow the segment would still carry power at its ends
    problem = LisaGB(n_sources=1, t_obs=2 * 31557600.0)
    p = jnp.array(
        [[problem.f0_range[1], problem.chirp_mass_range[1], 1e-22, 1.0, 0.3, 2.0, 0.5, 0.0]]
    )
    fdot = fdot_from_chirp_mass(p[:, 1], p[:, 0])
    segments = problem.response.get_tdi(
        p.at[:, 1].set(fdot), tdi_generation=1.5, tdi_combination="AET"
    )
    power = jnp.abs(jnp.stack(segments, axis=-1)[0, :, 0]) ** 2
    edge = (power[:8].sum() + power[-8:].sum()) / power.sum()
    assert float(edge) < 1e-3


def test_the_frequency_bin_width_resolves_the_annual_doppler_shift():
    # the doppler track across a year is what localises a source on the sky, so the
    # wdm frequency resolution has to be finer than its amplitude
    problem = LisaGB(n_sources=1, t_obs=2 * 31557600.0)
    signal = problem.clean_signal(
        jnp.array([[problem.f0_range[1], CHIRP_MASS, 1e-22, 1.0, 0.3, 2.0, 0.5, 0.0]])
    )
    n_freq = signal.shape[-2]
    nyquist = n_freq / (2.0 * problem.t_obs)
    band_width = nyquist / problem.wdm_freq_bands
    assert band_width < problem.f0_range[1] * V_OVER_C


def test_the_bin_width_does_not_resolve_the_doppler_shift_over_a_month():
    # a month spans ~1/12 of the orbit and the (n // k + 1) * k padding inflates the
    # band, so the doppler track is not resolvable at this baseline -- as expected
    problem = LisaGB(n_sources=1, t_obs=MONTH)
    signal = problem.clean_signal(
        jnp.array([[problem.f0_range[1], CHIRP_MASS, 1e-22, 1.0, 0.3, 2.0, 0.5, 0.0]])
    )
    n_freq = signal.shape[-2]
    nyquist = n_freq / (2.0 * problem.t_obs)
    band_width = nyquist / problem.wdm_freq_bands
    assert band_width > problem.f0_range[1] * V_OVER_C
