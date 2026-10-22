"""The sliding window covers the f0 it draws from, and response_points covers a source."""

import math

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from canna.lisa.physics import chirp_mass_support, fdot_from_chirp_mass, response_span
from canna.lisa.problem import WINDOW_STRIDE, LisaGB, f0_window, sample_f, window_start
from canna.lisa.simulator import clean_signal, preprocess
from ._helpers import window


jax.config.update("jax_enable_x64", True)

MONTH = 2592000.0
YEAR = 31557600.0
TWO_YEARS = 2 * YEAR
V_OVER_C = 29785.0 / 299792458.0
CHIRP_MASS = 0.3  # a typical double white dwarf [Msun]


def test_the_loudest_pixel_sits_at_the_injected_frequency():
    problem = LisaGB(n_sources=1, t_obs=MONTH)
    low, high = f0_window(problem, window(problem))
    f0 = float(jnp.sqrt(low * high))
    signal = clean_signal(
        problem,
        jnp.array([[f0, CHIRP_MASS, 1e-22, 1.0, 0.3, 2.0, 0.5, 0.0]]),
        window(problem),
    )

    # preprocess keeps one row per rfft bin of the window; A's power peaks at f0
    stream = preprocess(problem, signal, window(problem))
    bin_ = f0 * problem.t_obs - int(window_start(problem, window(problem)))
    loudest = int(jnp.argmax(stream[:, 0] ** 2 + stream[:, 3] ** 2))
    assert abs(loudest - bin_) < 2


def test_no_signal_energy_lives_outside_the_sources_response_segment():
    problem = LisaGB(n_sources=1, t_obs=MONTH)
    low, high = f0_window(problem, window(problem))
    f0 = float(jnp.sqrt(low * high))
    signal = clean_signal(
        problem,
        jnp.array([[f0, CHIRP_MASS, 1e-22, 1.0, 0.3, 2.0, 0.5, 0.0]]),
        window(problem),
    )
    power = jnp.abs(signal[:, 0]) ** 2
    start = int(
        problem.response.get_kmin(jnp.array([f0]))[0]
        - window_start(problem, window(problem))
    )
    inside = power[start : start + problem.response_points].sum()
    assert float((power.sum() - inside) / power.sum()) < 1e-12


def test_a_source_at_either_window_edge_keeps_its_response_inside_the_window():
    # f0_window pads the window by response_points/2 at both ends, so a source at either
    # edge of what the prior can draw lands its whole response_points-wide segment inside
    # the window instead of having power silently clipped by the scatter add
    problem = LisaGB(n_sources=1, t_obs=MONTH)
    for edge in f0_window(problem, window(problem)):
        kmin_edge = int(problem.response.get_kmin(jnp.atleast_1d(edge))[0])
        start = kmin_edge - int(window_start(problem, window(problem)))
        assert start >= 0
        assert start + problem.response_points <= problem.window_bins


def test_the_guard_band_is_a_small_fraction_of_the_window():
    # the two response_points/2 guards are the only part of the window no source can be
    # drawn at; keep that waste bounded. The window is 4 response_points wide
    # (WINDOW_RESPONSES), so the guards take exactly a quarter of it
    problem = LisaGB(n_sources=1, t_obs=MONTH)
    assert problem.response_points / problem.window_bins <= 0.25


def test_sample_f_snaps_to_a_window_start_that_window_start_recovers():
    problem = LisaGB(n_sources=1, t_obs=TWO_YEARS)
    f = jax.vmap(lambda k: sample_f(problem, k))(jr.split(jr.key(0), 512))
    start = jax.vmap(lambda f: window_start(problem, f))(f)
    lo, hi = problem.window_index_range
    assert np.all(start % WINDOW_STRIDE == 0)
    assert np.all((start >= lo * WINDOW_STRIDE) & (start <= hi * WINDOW_STRIDE))
    np.testing.assert_array_equal(f, (start + problem.window_bins / 2) / problem.t_obs)
    # a rounding nudge on f lands on the same window
    nudged = jax.vmap(lambda f: window_start(problem, f))(f * (1 + 1e-12))
    np.testing.assert_array_equal(nudged, start)


def test_response_points_covers_the_doppler_the_drift_and_the_antenna_sidebands():
    # the chirp only drifts upward, so half the window has to hold doppler + drift + 4/yr
    problem = LisaGB(n_sources=1, t_obs=TWO_YEARS)
    f0_hi = problem.f0_range[1]
    fdot_hi = fdot_from_chirp_mass(chirp_mass_support(problem.mass_range)[1], f0_hi)
    bin_width = 1.0 / problem.t_obs
    doppler_bins = f0_hi * V_OVER_C / bin_width
    drift_bins = fdot_hi * problem.t_obs / bin_width
    antenna_bins = 4 * problem.t_obs / YEAR
    assert problem.response_points / 2.0 > doppler_bins + drift_bins + antenna_bins


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
    mc, hi = chirp_mass_support(problem.mass_range)[1], problem.f0_range[1]
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
                chirp_mass_support(problem.mass_range)[1],
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
    # frequency resolution has to be finer than its amplitude
    problem = LisaGB(n_sources=1, t_obs=TWO_YEARS, f0_range=(3.0e-3, 12.0e-3))
    channel_width = 1 / problem.t_obs
    assert channel_width < float(f0_window(problem, window(problem))[0]) * V_OVER_C


def test_the_channel_width_does_not_resolve_the_doppler_shift_over_a_month():
    # a month spans ~1/12 of the orbit, so the doppler track is not resolvable at this
    # baseline -- as expected
    problem = LisaGB(n_sources=1, t_obs=MONTH)
    channel_width = 1 / problem.t_obs
    assert channel_width > float(f0_window(problem, window(problem))[1]) * V_OVER_C
