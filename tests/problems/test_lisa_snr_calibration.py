"""SNR is calibrated: the matched filter scales with amplitude and baseline as physics says."""

import jax
import jax.numpy as jnp
import jax.random as jr

from canna.problems import LisaGB

jax.config.update("jax_enable_x64", True)

MONTH = 2419200.0
YEAR = 31536000.0
SOURCE = jnp.array([[8.6e-3, 1e-16, 1e-22, 1.0, 0.3, 2.0, 0.5, 0.0]])


def test_snr_is_linear_in_amplitude():
    problem = LisaGB(n_sources=1, t_obs=MONTH)
    faint = problem.snr(SOURCE.at[:, 2].set(1e-23))
    loud = problem.snr(SOURCE.at[:, 2].set(1e-22))
    assert jnp.allclose(loud, 10.0 * faint, rtol=1e-6)


def test_snr_converges_as_the_response_grid_is_refined():
    # snr is a numerical integral over each source's local response window; refining
    # response_points from 8 to 32 must resolve it more finely without moving it.
    common = dict(
        n_sources=2,
        t_obs=1.0e6,
        sampling_step=0.25,
        wdm_freq_bands=32,
        patch_downsample=4,
        f0_range=(3.0e-3, 3.2e-3),
    )
    coarse = LisaGB(response_points=8, **common)
    fine = LisaGB(response_points=32, **common)
    p = coarse.sample_physical(jr.key(73))
    assert jnp.allclose(coarse.snr(p), fine.snr(p), rtol=0.1)


def test_snr_grows_as_the_square_root_of_the_baseline():
    # a monochromatic source accumulates snr^2 linearly in observation time
    one_year = LisaGB(n_sources=1, t_obs=YEAR).snr(SOURCE)
    two_years = LisaGB(n_sources=1, t_obs=2 * YEAR).snr(SOURCE)
    assert jnp.allclose(two_years, jnp.sqrt(2.0) * one_year, rtol=0.15)


def test_snr_of_a_loud_source_is_order_ten_over_a_month():
    # anchors the absolute normalisation: 1e-22 at 8.6 mHz is a marginally loud LISA gb
    snr = float(LisaGB(n_sources=1, t_obs=MONTH).snr(SOURCE))
    assert 3.0 < snr < 30.0


def test_clean_signal_holds_strain_density_not_dft_coefficients():
    # parseval: sum|hbar|^2 / t_obs = integral |hbar|^2 df = a^2 t_obs / 2 for a
    # monochromatic source, up to the O(1) tdi antenna response
    problem = LisaGB(n_sources=1, t_obs=MONTH)
    spectra = problem.clean_signal(SOURCE)
    energy = jnp.sum(jnp.abs(spectra[..., 0]) ** 2) / problem.t_obs
    implied_amplitude = jnp.sqrt(2.0 * energy / problem.t_obs)
    assert 0.05 < implied_amplitude / SOURCE[0, 2] < 2.0


def test_conditioning_image_of_pure_noise_has_order_unity_pixels():
    # preprocess whitens by the same variance sample_observation injects, so the
    # image is a compressed unit-variance field, not an arbitrary scale
    problem = LisaGB(n_sources=1, t_obs=MONTH)
    faint = SOURCE.at[:, 2].set(1e-30)
    image = problem.preprocess(problem.sample_observation(jr.key(0), faint))
    assert 0.2 < float(image.std()) < 5.0


def test_snr_is_the_signal_norm_in_noise_units():
    # monte-carlo: snr == sqrt(sum |clean|^2 / measured noise variance), summed over
    # the full two-sided hermitian band. mask on the analytic band snr uses -- noise_psd
    # is exactly 0 at f=0, so a measured-variance mask would admit dc float-dust
    problem = LisaGB(n_sources=3, t_obs=60000.0, wdm_freq_bands=128)
    p = problem.sample_physical(jr.key(9))
    clean = problem.clean_signal(p)
    n = clean.shape[-2]
    freqs = jnp.fft.fftfreq(n, problem.t_obs / n)
    band = problem.noise_psd(freqs) * problem.t_obs / 2.0 > 0.0

    def accumulate(total, key):
        residual = problem.sample_observation(key, p) - clean
        return total + jnp.abs(residual) ** 2, None

    summed, _ = jax.lax.scan(
        accumulate,
        jnp.zeros_like(clean, dtype=jnp.float64),
        jr.split(jr.key(20), 4096),
    )
    var = summed / 4096
    weighted = jnp.where(band, jnp.abs(clean) ** 2 / jnp.where(band, var, 1.0), 0.0)
    assert jnp.allclose(problem.snr(p), jnp.sqrt(jnp.sum(weighted)), rtol=0.03)
