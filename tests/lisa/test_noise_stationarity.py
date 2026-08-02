"""Noise intensity is stationary in time and independent of the observation baseline."""

import jax.numpy as jnp
import jax.random as jr
from einops import rearrange

from canna.lisa import LisaGB
from ._helpers import window


MONTH = 28 * 86400.0
FMAX = 8.0e-3  # common band, inside the nyquist of every baseline tested here
FAINT = jnp.array([[1.0e-4, 1.0e-18, 1.0e-30, 1.0, 0.3, 2.0, 0.5, 0.0]])


def noise_series(problem, key):
    # band-limit to FMAX so baselines with different nyquist are compared like for like,
    # and scale the ifft by n / t_obs so the series is a strain amplitude, not a
    # convention-dependent per-sample average
    spectrum = problem.sample_observation(key, FAINT, window(problem))
    n = spectrum.shape[-2]
    freqs = jnp.fft.fftfreq(n, problem.t_obs / n)
    kept = jnp.where((jnp.abs(freqs) <= FMAX)[:, None], spectrum, 0.0)
    return (jnp.fft.ifft(kept, axis=-2) * n / problem.t_obs).real


def test_noise_intensity_independent_of_observation_baseline():
    month = LisaGB(n_sources=1, t_obs=MONTH)
    year = LisaGB(n_sources=1, t_obs=12 * MONTH)
    month_std = noise_series(month, jr.key(0)).std(axis=0)
    year_std = noise_series(year, jr.key(1)).std(axis=0)
    assert jnp.allclose(year_std, month_std, rtol=0.05)


def test_first_month_of_a_year_matches_a_standalone_month():
    month = LisaGB(n_sources=1, t_obs=MONTH)
    year = LisaGB(n_sources=1, t_obs=12 * MONTH)
    month_std = noise_series(month, jr.key(2)).std(axis=0)
    series = noise_series(year, jr.key(3))
    first_month = series[: series.shape[0] // 12]
    assert jnp.allclose(first_month.std(axis=0), month_std, rtol=0.05)


def test_noise_intensity_stationary_across_the_months_of_a_year():
    year = LisaGB(n_sources=1, t_obs=12 * MONTH)
    series = noise_series(year, jr.key(4))
    blocks = rearrange(series[: (series.shape[0] // 12) * 12], "(m s) c -> m s c", m=12)
    per_month = blocks.std(axis=1)
    assert jnp.allclose(per_month, per_month.mean(axis=0), rtol=0.05)
