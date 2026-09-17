"""With the noise variance normalised as psd * t_obs / 2.0 (no sampling_step factor),
snr, sample_observation, and preprocess are all exactly invariant to sampling_step."""

import jax.numpy as jnp
import jax.random as jr

from canna.lisa import LisaGB
from ._helpers import window


def test_snr_invariant_to_sampling_step():
    p_key = jr.key(65)
    fine = LisaGB(n_sources=1, sampling_step=0.1)
    coarse = LisaGB(n_sources=1, sampling_step=10.0)
    f = window(fine)
    p = fine.sample_physical(p_key, f)
    assert float(fine.snr(p, f)) > 0.0
    assert jnp.allclose(fine.snr(p, f), coarse.snr(p, f), atol=1e-6, rtol=1e-6)


def test_sample_observation_noisy_invariant_to_sampling_step():
    # scale = sqrt(psd * t_obs / 2.0) carries no sampling_step term, and neither
    # clean_signal's shape nor noise_psd depends on sampling_step, so for a fixed key
    # and physical parameters the whole noisy draw is exactly invariant to it
    p_key = jr.key(67)
    obs_key = jr.key(68)
    fine = LisaGB(n_sources=1, sampling_step=0.1)
    coarse = LisaGB(n_sources=1, sampling_step=10.0)
    f = window(fine)
    p = fine.sample_physical(p_key, f)
    o_fine = fine.sample_observation(obs_key, p, f)
    o_coarse = coarse.sample_observation(obs_key, p, f)
    assert jnp.allclose(o_fine, o_coarse, atol=0.0, rtol=1e-12)


def test_preprocess_invariant_to_sampling_step():
    # wdm_transform/windows.py phi_window opens with `del nf, dt`, so the dt argument
    # preprocess forwards as `dt=self.sampling_step` is discarded before it can affect
    # the window, and the pre-WDM whitening power is `noise_psd(freqs) * t_obs / 2.0`
    # with no sampling_step term either. So preprocess's output is exactly identical
    # across sampling_step values.
    p_key = jr.key(50)
    fine = LisaGB(n_sources=1, sampling_step=1.0)
    coarse = LisaGB(n_sources=1, sampling_step=20.0)
    f = window(fine)
    p = fine.sample_physical(p_key, f)
    o = fine.clean_signal(p, f)
    img_fine = fine.preprocess(o, f)
    img_coarse = coarse.preprocess(o, f)
    assert jnp.allclose(img_fine, img_coarse, atol=1e-9, rtol=1e-9)
