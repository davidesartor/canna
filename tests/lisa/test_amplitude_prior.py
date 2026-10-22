"""The amplitude prior is an snr range: each drawn source has its own snr in snr_range."""

from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from canna.lisa.problem import (
    LisaGB,
    amplitude_unit,
    noise_psd,
    sky_averaged_sensitivity,
)
from canna.lisa.simulator import sample_physical, source_snrs
from ._helpers import window

jax.config.update("jax_enable_x64", True)

YEAR = 31557600.0


def test_drawn_sources_have_their_snr_in_range_loudest_first():
    problem = LisaGB(n_sources=4, f0_range=(1.2e-3, 1.6e-3))
    f = window(problem)
    draw = eqx.filter_jit(jax.vmap(partial(sample_physical, problem), in_axes=(0, None)))
    p = draw(jr.split(jr.key(0), 64), f)
    rho = source_snrs(problem, p, f)
    assert jnp.all(rho >= problem.snr_range[0] * (1 - 1e-6))
    assert jnp.all(rho <= problem.snr_range[1] * (1 + 1e-6))
    assert jnp.all(jnp.diff(rho, axis=-1) <= 1e-6 * rho[..., 1:])


def test_drawn_snrs_are_log_uniform():
    # 1/rho over [7, 1000]: log rho uniform, so its mean is the midpoint of the log range
    problem = LisaGB(n_sources=1, f0_range=(1.2e-3, 1.6e-3))
    f = window(problem)
    draw = eqx.filter_jit(jax.vmap(partial(sample_physical, problem), in_axes=(0, None)))
    p = draw(jr.split(jr.key(1), 512), f)
    log_rho = jnp.log(source_snrs(problem, p, f))
    lo, hi = jnp.log(jnp.asarray(problem.snr_range))
    assert jnp.allclose(jnp.mean(log_rho), (lo + hi) / 2, atol=0.25)
    assert jnp.allclose(jnp.std(log_rho), (hi - lo) / jnp.sqrt(12.0), rtol=0.1)


def test_amplitude_unit_tracks_the_sensitivity_across_the_band():
    # the unit follows the sensitivity curve, which falls steeply from 0.1 mHz, bottoms out
    # around 5-6 mHz where acceleration noise has died away and the arm-length transfer
    # function has not yet bitten, then turns back up
    problem = LisaGB(n_sources=1)
    band = (3.0e-4, 3.0e-3, 6.0e-3, 1.2e-2)
    unit = jnp.stack([amplitude_unit(problem, jnp.asarray(f0)) for f0 in band])
    assert unit[0] > unit[1] > unit[2], "must fall towards the sweet spot"
    assert unit[3] > unit[2], "and rise again past it"
    assert unit[0] / unit[2] > 100.0, "0.1-10 mHz is orders of magnitude, not a tweak"


def test_amplitude_unit_is_about_snr_one():
    # the chart's unit is the sky-averaged snr-1 amplitude, up to the O(1) sky and orientation factor
    problem = LisaGB(n_sources=1, f0_range=(1.2e-3, 1.6e-3))
    f = window(problem)
    draw = eqx.filter_jit(jax.vmap(partial(sample_physical, problem), in_axes=(0, None)))
    p = draw(jr.split(jr.key(2), 64), f)
    unit_snr = source_snrs(problem, p, f) / p[..., 2] * amplitude_unit(problem, f)
    assert 0.1 < float(jnp.median(unit_snr)) < 10.0


def test_amplitude_unit_scales_with_the_baseline():
    # rho grows as sqrt(t_obs), so the amplitude needed for a fixed rho falls the same way
    one = LisaGB(n_sources=1, t_obs=YEAR, f0_range=(1.0e-3, 12.0e-3))
    two = LisaGB(n_sources=1, t_obs=2 * YEAR, f0_range=(1.0e-3, 12.0e-3))
    f = jnp.asarray(3.0e-3)
    assert jnp.allclose(
        amplitude_unit(two, f), amplitude_unit(one, f) / jnp.sqrt(2.0), rtol=1e-6
    )


def test_sensitivity_and_noise_psd_share_their_noise_levels():
    # both are built from single_link_noise, so raising a noise level must move both
    quiet = LisaGB(n_sources=1, oms_noise=15.0, acceleration_noise=3.0)
    loud = LisaGB(n_sources=1, oms_noise=30.0, acceleration_noise=3.0)
    f = jnp.asarray(1.0e-2)  # optical-metrology dominated, so oms_noise sets the scale
    assert sky_averaged_sensitivity(loud, f) > 3.5 * sky_averaged_sensitivity(quiet, f)
    assert noise_psd(loud, f)[0] > 3.5 * noise_psd(quiet, f)[0]
