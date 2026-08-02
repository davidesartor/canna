"""preprocess whitens by noise_psd before the WDM transform; that whitening
(noise_psd(f) * t_obs / 2) does not depend on sampling_step."""

import jax
import jax.numpy as jnp
import jax.random as jr

from canna.lisa import LisaGB
from ._helpers import window



def test_whitening_flattens_the_noise_to_unit_variance():
    # the whitening power is noise_psd(f) * t_obs / 2, so dividing the injected
    # noise by its sqrt leaves a unit-variance field
    problem = LisaGB(
        n_sources=3, t_obs=1.0e6, wdm_freq_bands=64, f0_range=(3.0e-3, 3.2e-3)
    )
    p = problem.sample_physical(jr.key(0), window(problem))
    clean = problem.clean_signal(p, window(problem))
    power = problem.noise_psd(problem.window_freqs(window(problem))) * problem.t_obs / 2.0
    whiten = jax.lax.rsqrt(power)

    def accumulate(total, key):
        whitened = (problem.sample_observation(key, p, window(problem)) - clean) * whiten
        return total + jnp.abs(whitened) ** 2, None

    summed, _ = jax.lax.scan(
        accumulate,
        jnp.zeros_like(clean, dtype=jnp.float64),
        jr.split(jr.key(30), 4096),
    )
    assert jnp.allclose(jnp.mean(summed / 4096), 1.0, atol=0.05)
