"""The sliding window covers the f0 it draws from, and response_points covers a source."""

import math

import jax
import jax.numpy as jnp
import pytest

from canna.lisa import LisaGB
from canna.lisa.priors import fdot_from_chirp_mass
from canna.lisa.problem import response_span
from ._helpers import window


jax.config.update("jax_enable_x64", True)

MONTH = 2592000.0
TWO_YEARS = 2 * 31557600.0
V_OVER_C = 29785.0 / 299792458.0
CHIRP_MASS = 0.3  # a typical double white dwarf [Msun]


def test_the_loudest_pixel_sits_at_the_injected_frequency():
    problem = LisaGB(n_sources=1, t_obs=MONTH)
    low, high = problem.f0_window(window(problem))
    f0 = float(jnp.sqrt(low * high))
    signal = problem.clean_signal(
        jnp.array([[f0, CHIRP_MASS, 1e-22, 1.0, 0.3, 2.0, 0.5, 0.0]]), window(problem)
    )

    # each wdm frequency channel is wdm_times // 2 rfft bins of the window
    image = problem.preprocess(signal, window(problem))
    channel = (f0 * problem.t_obs - int(problem.window_start(window(problem)))) / (
        problem.wdm_times / 2
    )
    loudest = int(jnp.argmax(jnp.abs(image[..., 0]).max(axis=0)))
    assert abs(loudest - channel) < 2


def test_no_signal_energy_lives_outside_the_sources_response_segment():
    problem = LisaGB(n_sources=1, t_obs=MONTH)
    low, high = problem.f0_window(window(problem))
    f0 = float(jnp.sqrt(low * high))
    signal = problem.clean_signal(
        jnp.array([[f0, CHIRP_MASS, 1e-22, 1.0, 0.3, 2.0, 0.5, 0.0]]), window(problem)
    )
    power = jnp.abs(signal[:, 0]) ** 2
    start = int(
        problem.response.get_kmin(jnp.array([f0]))[0] - problem.window_start(window(problem))
    )
    inside = power[start : start + problem.response_points].sum()
    assert float((power.sum() - inside) / power.sum()) < 1e-12


def test_a_source_at_either_window_edge_keeps_its_response_inside_the_window():
    # f0_window pads the window by response_points/2 at both ends, so a source at either
    # edge of what the prior can draw lands its whole response_points-wide segment inside
    # the window instead of having power silently clipped by the scatter add
    problem = LisaGB(n_sources=1, t_obs=MONTH)
    for edge in problem.f0_window(window(problem)):
        kmin_edge = int(problem.response.get_kmin(jnp.atleast_1d(edge))[0])
        start = kmin_edge - int(problem.window_start(window(problem)))
        assert start >= 0
        assert start + problem.response_points <= problem.window_bins


def test_the_guard_band_is_a_small_fraction_of_the_window():
    # the two response_points/2 guards are the only part of the window no source can be
    # drawn at; keep that waste bounded
    problem = LisaGB(n_sources=1, t_obs=MONTH)
    assert problem.response_points / problem.window_bins < 0.2


def test_response_points_covers_the_annual_doppler_and_the_fdot_drift():
    # the chirp only drifts upward, so half the window has to hold doppler + drift
    problem = LisaGB(n_sources=1, t_obs=TWO_YEARS)
    f0_hi, fdot_hi = problem.f0_range[1], problem.fdot_range[1]
    bin_width = 1.0 / problem.t_obs
    doppler_bins = f0_hi * V_OVER_C / bin_width
    drift_bins = fdot_hi * problem.t_obs / bin_width
    assert problem.response_points / 2.0 > doppler_bins + drift_bins


def test_a_response_window_narrower_than_the_chirp_caps_the_f0_prior():
    # the loudest chirp the prior allows sits at the top of the f0 band with the
    # heaviest pair; at a two-year baseline it needs far more than 256 bins, so a
    # 256-bin budget lowers f0_range[1] to the band it can hold instead of failing
    asked = (1e-4, 12.0e-3)
    with pytest.warns(UserWarning, match="response_points=256"):
        problem = LisaGB(
            n_sources=1, t_obs=TWO_YEARS, response_points=256, f0_range=asked
        )

    assert problem.f0_range[0] == asked[0]
    assert problem.f0_range[1] < asked[1]
    assert problem.response_points == 256

    # the cap is tight: the top of the capped band fills the window exactly, and the
    # next representable f0 above it would already overrun it
    mc, hi = problem.chirp_mass_range[1], problem.f0_range[1]
    assert response_span(hi, mc, TWO_YEARS) == 256
    assert response_span(math.nextafter(hi, 1.0), mc, TWO_YEARS) > 256


def test_a_budget_leaving_no_band_at_all_is_rejected_at_init():
    # capping can only lower f0_range[1]; if it lands under f0_range[0] there is no
    # prior left to sample, and that is an error rather than a warning
    with pytest.raises(AssertionError, match="no band at all"):
        LisaGB(
            n_sources=1,
            t_obs=TWO_YEARS,
            response_points=256,
            f0_range=(9.0e-3, 12.0e-3),
        )


def test_an_ample_response_window_leaves_the_f0_prior_alone():
    # the budget only ever binds from above: one wider than the band asks for nothing
    asked = (1e-4, 12.0e-3)
    problem = LisaGB(
        n_sources=1, t_obs=TWO_YEARS, response_points=1024, f0_range=asked
    )
    assert problem.f0_range == asked


def test_the_local_response_segment_decays_to_its_edges():
    # if response_points were too narrow the segment would still carry power at its ends
    problem = LisaGB(n_sources=1, t_obs=TWO_YEARS)
    p = jnp.array(
        [
            [
                problem.f0_range[1],
                problem.chirp_mass_range[1],
                1e-22,
                1.0,
                0.3,
                2.0,
                0.5,
                0.0,
            ]
        ]
    )
    fdot = fdot_from_chirp_mass(p[:, 1], p[:, 0])
    segments = problem.response.get_tdi(
        p.at[:, 1].set(fdot), tdi_generation=1.5, tdi_combination="AET"
    )
    power = jnp.abs(jnp.stack(segments, axis=-1)[0, :, 0]) ** 2
    edge = (power[:8].sum() + power[-8:].sum()) / power.sum()
    assert float(edge) < 1e-3


def test_the_frequency_channel_width_resolves_the_annual_doppler_shift():
    # the doppler track across a year is what localises a source on the sky, so the
    # wdm frequency resolution has to be finer than its amplitude
    problem = LisaGB(n_sources=1, t_obs=TWO_YEARS, f0_range=(3.0e-3, 12.0e-3))
    channel_width = problem.wdm_times / 2 / problem.t_obs
    assert channel_width < float(problem.f0_window(window(problem))[0]) * V_OVER_C


def test_the_channel_width_does_not_resolve_the_doppler_shift_over_a_month():
    # a month spans ~1/12 of the orbit, so the doppler track is not resolvable at this
    # baseline -- as expected
    problem = LisaGB(n_sources=1, t_obs=MONTH)
    channel_width = problem.wdm_times / 2 / problem.t_obs
    assert channel_width > float(problem.f0_window(window(problem))[1]) * V_OVER_C
