"""preprocess whitens by noise_psd before the WDM transform; that whitening
(noise_psd(f) * t_obs / 2) does not depend on sampling_step."""

import jax
import jax.numpy as jnp
import jax.random as jr

from canna.problems import LisaGB


def test_whitening_flattens_the_noise_to_unit_variance():
    # the whitening power is noise_psd(f) * t_obs / 2, so dividing the injected
    # noise by its sqrt leaves a unit-variance field
    problem = LisaGB(n_sources=3, t_obs=60000.0, wdm_freq_bands=128)
    p = problem.sample_physical(jr.key(0))
    clean = problem.clean_signal(p)
    n = clean.shape[-2]
    freqs = jnp.fft.fftfreq(n, problem.t_obs / n)
    power = problem.noise_psd(freqs) * problem.t_obs / 2.0
    whiten = jnp.where(power > 0.0, jax.lax.rsqrt(power), 1.0)

    def accumulate(total, key):
        whitened = (problem.sample_observation(key, p) - clean) * whiten
        return total + jnp.abs(whitened) ** 2, None

    summed, _ = jax.lax.scan(
        accumulate,
        jnp.zeros_like(clean, dtype=jnp.float64),
        jr.split(jr.key(30), 4096),
    )
    band = power > 0.0
    assert jnp.allclose(jnp.mean((summed / 4096)[band]), 1.0, atol=0.05)
