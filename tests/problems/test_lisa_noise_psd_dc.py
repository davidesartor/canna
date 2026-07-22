"""noise_psd is inf at DC so whitening (rsqrt -> 0) and SNR (1/psd -> 0) drop the
zero-frequency bin automatically, while staying finite and positive everywhere else."""

import jax
import jax.numpy as jnp

from canna.problems import LisaGB


def test_noise_psd_is_inf_at_dc_and_finite_positive_elsewhere():
    problem = LisaGB(n_sources=1, t_obs=60000.0, f0_range=(5.0e-3, 1.0e-2), wdm_freq_bands=128)
    freqs = jnp.array([0.0, 1.0e-3, 5.0e-3, 1.0e-2])
    psd = problem.noise_psd(freqs)  # (4, 3)
    assert jnp.all(jnp.isinf(psd[0]))  # every channel inf at DC
    assert jnp.all(jnp.isfinite(psd[1:]))
    assert jnp.all(psd[1:] > 0.0)


def test_dc_whitens_to_zero_without_nan():
    problem = LisaGB(n_sources=1, t_obs=60000.0, f0_range=(5.0e-3, 1.0e-2), wdm_freq_bands=128)
    power = problem.noise_psd(jnp.array([0.0, 5.0e-3])) * problem.t_obs / 2.0
    whitened = jax.lax.rsqrt(power)  # rsqrt(inf) == 0, no guard needed
    assert jnp.all(whitened[0] == 0.0)
    assert jnp.all(jnp.isfinite(whitened[1]) & (whitened[1] > 0.0))
