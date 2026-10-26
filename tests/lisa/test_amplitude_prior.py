"""The amplitude comes from the (f0, fdot, A) prior; the chart's amplitude unit."""

from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from canna.lisa.physics import SOURCE_COVARIANCE, SOURCE_MEAN, SOURCE_RESIDUAL
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


def test_drawn_amplitudes_follow_the_catalogue_not_the_snr():
    # log10 A given f0 is the prior's conditional: about -22.9 +- 0.25 at 1.4 mHz, whatever the noise
    problem = LisaGB(n_sources=1, f0_range=(1.2e-3, 1.6e-3))
    f = window(problem)
    draw = eqx.filter_jit(jax.vmap(partial(sample_physical, problem), in_axes=(0, None)))
    p = draw(jr.split(jr.key(1), 2048), f)[:, 0]
    slope = SOURCE_COVARIANCE[0, 2] / SOURCE_COVARIANCE[0, 0]
    mean = SOURCE_MEAN[2] + slope * (jnp.log10(p[:, 0]) - SOURCE_MEAN[0])
    residual = jnp.log10(p[:, 2]) - mean
    assert abs(float(residual.mean())) < 0.02
    assert jnp.allclose(residual.std(), jnp.sqrt(SOURCE_RESIDUAL[1, 1]), rtol=0.05)


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
