"""fdot is a drawn column: clean_signal chirps by it, and the prior keeps it positive."""

from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from canna.lisa.physics import log10_fdot_mean
from canna.lisa.problem import LisaGB, f0_window
from canna.lisa.simulator import clean_signal, sample_physical
from ._helpers import window


jax.config.update("jax_enable_x64", True)

TWO_YEARS = 2 * 365.25 * 24 * 60 * 60


def test_the_prior_chirp_grows_with_frequency_as_radiation_reaction_does():
    # fdot ~ f0^(11/3) at fixed chirp mass; the catalogue's slope, 3.48 per dex, sits just below it
    slope = float(log10_fdot_mean(1e-2) - log10_fdot_mean(1e-3))
    assert 3.3 < slope < 11 / 3


def test_a_verification_binary_sits_inside_the_prior():
    # HM Cnc (f0 = 6.22 mHz) chirps at 3.6e-16 Hz/s: 0.21 dex, about a sigma, below the prior's mean there
    log_fdot = float(log10_fdot_mean(6.22e-3))
    assert abs(log_fdot - jnp.log10(3.6e-16)) < 0.3


def test_the_fdot_column_drifts_the_waveform_by_the_predicted_amount():
    # the band's power-weighted centre sits at f0 + fdot * t_obs / 2, so the shift
    # between two chirps pins down the fdot clean_signal actually used
    # 1024 bins hold the 75-bin doppler and the 60-bin drift whole
    problem = LisaGB(n_sources=1, t_obs=TWO_YEARS, response_points=1024, f0_range=(9.5e-3, 1.0e-2))
    low, high = f0_window(problem, window(problem))
    p = jnp.array([[float(jnp.sqrt(low * high)), 0.0, 1e-22, 1.0, 0.3, 2.0, 0.5, 0.0]])

    def centre(fdot: float) -> float:
        signal = clean_signal(problem, p.at[0, 1].set(fdot), window(problem))
        power = jnp.abs(signal[:, 0]) ** 2
        return float(jnp.sum(power * jnp.arange(power.size)) / jnp.sum(power))

    slow, fast = 1e-16, 1.5e-14
    shift = centre(fast) - centre(slow)
    predicted = (fast - slow) * problem.t_obs**2 / 2
    assert abs(shift - predicted) < 0.05 * predicted


def test_drawn_chirps_span_decades_across_the_band():
    problem = LisaGB(n_sources=1)
    draw = eqx.filter_jit(jax.vmap(partial(sample_physical, problem), in_axes=(0, None)))
    p = draw(jr.split(jr.key(1), 256), window(problem)).reshape(-1, 8)
    assert jnp.all(p[:, 1] > 0)
    # radiation reaction is many orders of magnitude weaker at the bottom of the band
    lo, hi = (10.0 ** float(log10_fdot_mean(f0)) for f0 in problem.f0_range)
    assert lo < 1e-20 < 1e-15 < hi
