"""With the noise variance normalised as psd * t_obs / 2.0 (no sampling_step factor),
snr, sample_observation, and preprocess are all exactly invariant to sampling_step,
and snr is additionally invariant to the wdm_freq_bands/patch_downsample band-crop."""

import jax.numpy as jnp
import jax.random as jr

from canna.problems import LisaGB


def test_snr_invariant_to_sampling_step():
    p_key = jr.key(65)
    fine = LisaGB(n_sources=1, sampling_step=0.1)
    coarse = LisaGB(n_sources=1, sampling_step=10.0)
    p = fine.sample_physical(p_key)
    assert jnp.allclose(fine.snr(p), coarse.snr(p), atol=1e-6, rtol=1e-6)


def test_snr_invariant_to_wdm_band_crop_parameters():
    # snr sums |clean_signal|**2 / var over the *whole* padded F axis; the padding
    # introduced by a smaller wdm_freq_bands*patch_downsample product carries zero
    # signal (clean_signal only ever writes response_points bins around each
    # source's kmin), so it should not move the sum even though the two problems'
    # clean_signal outputs have entirely different shapes.
    p_key = jr.key(64)
    narrow = LisaGB(n_sources=1, wdm_freq_bands=4096, patch_downsample=8)
    wide = LisaGB(n_sources=1, wdm_freq_bands=512, patch_downsample=4)
    p = narrow.sample_physical(p_key)
    assert narrow.clean_signal(p).shape != wide.clean_signal(p).shape
    assert jnp.allclose(narrow.snr(p), wide.snr(p), atol=1e-6, rtol=1e-6)


def test_sample_observation_noisy_invariant_to_sampling_step():
    # scale = sqrt(psd * t_obs / 2.0) no longer carries a sampling_step term, and
    # neither clean_signal's shape nor noise_psd depends on sampling_step (see
    # test_clean_signal_frequency_axis_independent_of_sampling_step in
    # test_lisa_physical_sanity.py), so for a fixed key and physical parameters the
    # whole noisy draw should now be exactly invariant to sampling_step.
    p_key = jr.key(67)
    obs_key = jr.key(68)
    fine = LisaGB(n_sources=1, sampling_step=0.1)
    coarse = LisaGB(n_sources=1, sampling_step=10.0)
    p = fine.sample_physical(p_key)
    o_fine = fine.sample_observation(obs_key, p, clean=False)
    o_coarse = coarse.sample_observation(obs_key, p, clean=False)
    assert jnp.allclose(o_fine, o_coarse, atol=1e-9, rtol=1e-9)


def test_preprocess_invariant_to_sampling_step():
    # wdm_transform/windows.py phi_window opens with `del nf, dt`, so the dt argument
    # preprocess forwards as `dt=self.sampling_step` is discarded before it can affect
    # the window, and the pre-WDM whitening power is `noise_psd(freqs) * t_obs / 2.0`
    # with no sampling_step term either. So preprocess's output is exactly identical
    # across sampling_step values.
    p_key = jr.key(50)
    fine = LisaGB(n_sources=1, sampling_step=1.0)
    coarse = LisaGB(n_sources=1, sampling_step=20.0)
    p = fine.sample_physical(p_key)
    o = fine.clean_signal(p)
    img_fine = fine.preprocess(o)
    img_coarse = coarse.preprocess(o)
    assert jnp.allclose(img_fine, img_coarse, atol=1e-9, rtol=1e-9)
