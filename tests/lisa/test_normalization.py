"""With the noise variance normalised as psd * t_obs / 2.0 (no sampling_step factor),
snr, sample_observation, and preprocess are all exactly invariant to sampling_step, and
snr is additionally invariant to how wide a window wdm_freq_bands makes."""

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


def test_snr_invariant_to_the_window_width():
    # snr sums |clean_signal|**2 / var over the whole window; a wider window only adds
    # bins the source writes nothing into (clean_signal only ever fills response_points
    # bins around each source's kmin), so it must not move the sum. both windows start
    # at the same index, so the wide one strictly contains the narrow one
    common = dict(n_sources=1, t_obs=1.0e6, f0_range=(3.0e-3, 3.2e-3))
    narrow = LisaGB(wdm_freq_bands=64, **common)
    wide = LisaGB(wdm_freq_bands=128, **common)
    f_narrow, f_wide = window(narrow), window(wide)
    assert int(narrow.window_start(f_narrow)) == int(wide.window_start(f_wide))

    p = jnp.array([[3.1e-3, 0.3, 1e-22, 1.0, 0.3, 2.0, 0.5, 0.0]])
    assert narrow.clean_signal(p, f_narrow).shape != wide.clean_signal(p, f_wide).shape
    assert float(narrow.snr(p, f_narrow)) > 0.0
    assert jnp.allclose(narrow.snr(p, f_narrow), wide.snr(p, f_wide), rtol=1e-6)


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
