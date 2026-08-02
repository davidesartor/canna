"""The sliding window never reaches DC, so whitening is finite and positive across it --
which is what keeps the pipeline away from noise_psd's undefined f=0 point."""

import jax
import jax.numpy as jnp

from canna.lisa import LisaGB
from ._helpers import window


CONFIG = dict(n_sources=1, t_obs=2.0e6, f0_range=(5.0e-3, 1.0e-2), wdm_freq_bands=128)


def test_the_window_never_reaches_dc():
    problem = LisaGB(**CONFIG)
    assert int(problem.window_start(window(problem))) > 0
    assert float(problem.window_freqs(window(problem)).min()) > 0.0


def test_the_whitening_power_is_finite_and_positive_across_the_window():
    problem = LisaGB(**CONFIG)
    power = problem.noise_psd(problem.window_freqs(window(problem))) * problem.t_obs / 2.0
    whitened = jax.lax.rsqrt(power)
    assert jnp.all(jnp.isfinite(power) & (power > 0.0))
    assert jnp.all(jnp.isfinite(whitened) & (whitened > 0.0))


def test_noise_psd_is_finite_and_positive_across_the_measurement_band():
    problem = LisaGB(**CONFIG)
    psd = problem.noise_psd(jnp.array([1.0e-4, 1.0e-3, 5.0e-3, 1.0e-2]))
    assert jnp.all(jnp.isfinite(psd))
    assert jnp.all(psd > 0.0)


def test_noise_psd_is_undefined_exactly_at_dc():
    # the divergent acceleration term meets the 4 sin^2(omega L) TDI prefactor's zero,
    # so f=0 is inf * 0; the window_start > 0 invariant is what keeps it unreachable
    problem = LisaGB(**CONFIG)
    assert jnp.all(jnp.isnan(problem.noise_psd(jnp.array(0.0))))
