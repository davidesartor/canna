"""The amplitude prior is an snr range: drawing at snr_range really gives that snr."""

import jax
import jax.numpy as jnp
import jax.random as jr

from canna.lisa import LisaGB
from canna.lisa.problem import SKY_AVERAGED_SNR_SQUARED
from ._helpers import window

jax.config.update("jax_enable_x64", True)

YEAR = 31557600.0


def sky_averaged_snr(problem, f0, amplitude=1.0, n=512, key=0):
    """Monte-Carlo sqrt(<rho^2>) over sky, polarisation, inclination and phase."""
    keys = jr.split(jr.key(key), 5)
    shape = (n, 1)
    p = jnp.concatenate(
        [
            jnp.full(shape, f0),
            jnp.full(shape, 0.45),  # chirp mass; rho does not depend on it
            jnp.full(shape, amplitude),
            jr.uniform(keys[0], shape, maxval=2 * jnp.pi),
            jnp.arcsin(jr.uniform(keys[1], shape, minval=-1.0, maxval=1.0)),
            jr.uniform(keys[2], shape, maxval=2 * jnp.pi),
            jnp.arcsin(jr.uniform(keys[3], shape, minval=-1.0, maxval=1.0)),
            jr.uniform(keys[4], shape, maxval=2 * jnp.pi),
        ],
        axis=-1,
    )
    f = jnp.asarray(f0)
    squared = jax.lax.map(lambda row: problem.snr(row[None, :], f) ** 2, p)
    return float(jnp.sqrt(jnp.mean(squared)))


def test_sky_averaged_snr_constant_matches_the_response():
    # this is where SKY_AVERAGED_SNR_SQUARED comes from: rho^2 for a unit-amplitude
    # sky-averaged source is that constant times t_obs / sensitivity. Re-derive it rather
    # than trusting the literal, and pin it where the RCL approximation is still good
    for f0 in (3.0e-4, 1.0e-3, 1.40457e-3):
        problem = LisaGB(n_sources=1, wdm_freq_bands=64, f0_range=(f0 * 0.9, f0 * 1.1))
        measured = sky_averaged_snr(problem, f0) ** 2
        predicted = (
            SKY_AVERAGED_SNR_SQUARED
            * problem.t_obs
            / problem.sky_averaged_sensitivity(jnp.asarray(f0))
        )
        assert jnp.allclose(measured, predicted, rtol=0.05), (f0, measured, predicted)


def test_amplitude_window_delivers_the_requested_snr():
    # the point of the prior: a source drawn at the faint edge really is an snr ~7 source
    problem = LisaGB(n_sources=1, wdm_freq_bands=64, f0_range=(1.2e-3, 1.6e-3))
    f = window(problem)
    f0 = float(jnp.sqrt(jnp.prod(jnp.stack(problem.f0_window(f)))))
    low, high = problem.a_window(f)
    assert jnp.allclose(sky_averaged_snr(problem, f0, float(low)), 7.0, rtol=0.1)
    assert jnp.allclose(sky_averaged_snr(problem, f0, float(high)), 1000.0, rtol=0.1)


def test_amplitude_window_tracks_the_sensitivity_across_the_band():
    # a fixed amplitude box would be flat; this one follows the sensitivity curve, which
    # falls steeply from 0.1 mHz, bottoms out around 5-6 mHz where acceleration noise has
    # died away and the arm-length transfer function has not yet bitten, then turns back up
    problem = LisaGB(n_sources=1)
    band = (3.0e-4, 3.0e-3, 6.0e-3, 1.2e-2)
    faint = jnp.stack([problem.a_window(jnp.asarray(f0))[0] for f0 in band])
    assert faint[0] > faint[1] > faint[2], "must fall towards the sweet spot"
    assert faint[3] > faint[2], "and rise again past it"
    assert faint[0] / faint[2] > 100.0, "0.1-10 mHz is orders of magnitude, not a tweak"
    # the two edges keep the snr_range ratio at every frequency
    ratios = jnp.stack(
        [problem.a_window(jnp.asarray(f0))[1] / problem.a_window(jnp.asarray(f0))[0]
         for f0 in band]
    )
    assert jnp.allclose(ratios, problem.snr_range[1] / problem.snr_range[0], rtol=1e-6)


def test_amplitude_window_scales_with_the_baseline():
    # rho grows as sqrt(t_obs), so the amplitude needed for a fixed rho falls the same way
    one = LisaGB(n_sources=1, t_obs=YEAR, f0_range=(1.0e-3, 12.0e-3))
    two = LisaGB(n_sources=1, t_obs=2 * YEAR, f0_range=(1.0e-3, 12.0e-3))
    f = jnp.asarray(3.0e-3)
    assert jnp.allclose(
        jnp.stack(two.a_window(f)),
        jnp.stack(one.a_window(f)) / jnp.sqrt(2.0),
        rtol=1e-6,
    )


def test_sensitivity_and_noise_psd_share_their_noise_levels():
    # both are built from single_link_noise, so raising a noise level must move both
    quiet = LisaGB(n_sources=1, oms_noise=15.0, acceleration_noise=3.0)
    loud = LisaGB(n_sources=1, oms_noise=30.0, acceleration_noise=3.0)
    f = jnp.asarray(1.0e-2)  # optical-metrology dominated, so oms_noise sets the scale
    assert loud.sky_averaged_sensitivity(f) > 3.5 * quiet.sky_averaged_sensitivity(f)
    assert loud.noise_psd(f)[0] > 3.5 * quiet.noise_psd(f)[0]
