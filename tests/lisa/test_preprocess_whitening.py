"""preprocess whitens by noise_psd(f) * t_obs / 2 to unit variance in each real and imaginary part."""

import jax
import jax.numpy as jnp
import jax.random as jr

from canna.lisa.problem import LisaGB, noise_psd, window_freqs
from canna.lisa.simulator import (
    clean_signal,
    preprocess,
    sample_observation,
    sample_physical,
)
from ._helpers import window


def test_whitening_flattens_the_noise_to_unit_variance():
    # the whitening power is noise_psd(f) * t_obs / 2, so dividing the injected
    # noise by its sqrt leaves a unit-variance field
    problem = LisaGB(n_sources=3, t_obs=1.0e6, f0_range=(3.0e-3, 3.2e-3))
    f = window(problem)
    p = sample_physical(problem, jr.key(0), f)
    clean = clean_signal(problem, p, f)
    power = noise_psd(problem, window_freqs(problem, f)) * problem.t_obs / 2.0
    whiten = jax.lax.rsqrt(power)

    def accumulate(total, key):
        whitened = (sample_observation(problem, key, p, f) - clean) * whiten
        return total + jnp.abs(whitened) ** 2, None

    summed, _ = jax.lax.scan(
        accumulate,
        jnp.zeros_like(clean, dtype=jnp.float64),
        jr.split(jr.key(30), 4096),
    )
    assert jnp.allclose(jnp.mean(summed / 4096), 1.0, atol=0.05)


def test_preprocess_gives_unit_noise_in_every_part():
    # a source-free observation: preprocess must leave each of the six channels (A, E, T,
    # real and imaginary) at unit variance, which is what the PLE's edges are placed for
    problem = LisaGB(n_sources=1, t_obs=1.0e6, f0_range=(3.0e-3, 3.2e-3))
    f = window(problem)
    p = sample_physical(problem, jr.key(0), f).at[:, 2].set(0.0)  # zero amplitude
    stream = jax.vmap(
        lambda k: preprocess(problem, sample_observation(problem, k, p, f), f)
    )(jr.split(jr.key(31), 512))
    assert stream.shape[-1] == 6
    assert jnp.allclose(jnp.var(stream, axis=(0, 1)), 1.0, atol=0.05)
    assert jnp.allclose(jnp.mean(stream, axis=(0, 1)), 0.0, atol=0.05)
