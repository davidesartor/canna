"""SNR is calibrated: the matched filter scales with amplitude and baseline as physics says."""

import jax
import jax.numpy as jnp
import jax.random as jr

from canna.lisa.problem import LisaGB, f0_window
from canna.lisa.simulator import (
    clean_signal,
    preprocess,
    sample_observation,
    sample_physical,
    snr,
)
from ._helpers import window

jax.config.update("jax_enable_x64", True)

MONTH = 2419200.0
YEAR = 31536000.0
# a band the window can sit on whole, so the zero context brackets 8.5 mHz
BAND = (8.0e-3, 9.0e-3)
FDOT = 1e-17  # a typical double white dwarf's chirp at a few mHz [Hz/s]


def source_in(problem: LisaGB, amplitude: float = 1e-22) -> jnp.ndarray:
    """One source at the geometric centre of the window the zero context selects."""
    low, high = f0_window(problem, window(problem))
    f0 = float(jnp.sqrt(low * high))
    return jnp.array([[f0, FDOT, amplitude, 1.0, 0.3, 2.0, 0.5, 0.0]])


def test_snr_is_linear_in_amplitude():
    problem = LisaGB(n_sources=1, t_obs=MONTH, f0_range=BAND)
    faint = snr(problem, source_in(problem, 1e-23), window(problem))
    loud = snr(problem, source_in(problem, 1e-22), window(problem))
    assert float(faint) > 0.0
    assert jnp.allclose(loud, 10.0 * faint, rtol=1e-6)


def test_snr_converges_as_the_response_grid_is_refined():
    # snr is a numerical integral over each source's local response window; refining
    # response_points from 256 to 512 must resolve it more finely without moving it
    common = dict(
        n_sources=1,
        t_obs=1.0e6,
        f0_range=(3.0e-3, 3.2e-3),
    )
    coarse = LisaGB(response_points=256, **common)
    fine = LisaGB(response_points=512, **common)
    p = jnp.array([[3.1e-3, FDOT, 1e-22, 1.0, 0.3, 2.0, 0.5, 0.0]])
    assert jnp.allclose(
        snr(coarse, p, window(coarse)), snr(fine, p, window(fine)), rtol=0.1
    )


def test_snr_grows_as_the_square_root_of_the_baseline():
    # a monochromatic source accumulates snr^2 linearly in observation time
    one_year = LisaGB(n_sources=1, t_obs=YEAR, f0_range=BAND)
    two_years = LisaGB(n_sources=1, t_obs=2 * YEAR, f0_range=BAND)

    # the longer baseline has the narrower window, so its centre sits inside both
    source = source_in(two_years)
    low, high = f0_window(one_year, window(one_year))
    assert low < source[0, 0] < high
    assert jnp.allclose(
        snr(two_years, source, window(two_years)),
        jnp.sqrt(2.0) * snr(one_year, source, window(one_year)),
        rtol=0.05,
    )


def test_snr_of_a_loud_source_is_order_ten_over_a_month():
    # anchors the absolute normalisation: 1e-22 at 8.5 mHz is a marginally loud LISA gb
    problem = LisaGB(n_sources=1, t_obs=MONTH, f0_range=BAND)
    rho = float(snr(problem, source_in(problem), window(problem)))
    assert 3.0 < rho < 30.0


def test_clean_signal_holds_strain_density_not_dft_coefficients():
    # parseval: sum|hbar|^2 / t_obs = integral |hbar|^2 df = a^2 t_obs / 2 for a
    # monochromatic source, up to the O(1) tdi antenna response
    problem = LisaGB(n_sources=1, t_obs=MONTH, f0_range=BAND)
    source = source_in(problem)
    spectra = clean_signal(problem, source, window(problem))
    energy = jnp.sum(jnp.abs(spectra[..., 0]) ** 2) / problem.t_obs
    implied_amplitude = jnp.sqrt(2.0 * energy / problem.t_obs)
    assert 0.05 < implied_amplitude / source[0, 2] < 2.0


def test_conditioning_image_of_pure_noise_has_order_unity_pixels():
    # preprocess whitens by the same variance sample_observation injects, so the
    # image is a compressed unit-variance field, not an arbitrary scale
    problem = LisaGB(n_sources=1, t_obs=MONTH, f0_range=BAND)
    faint = source_in(problem, 1e-30)
    image = preprocess(
        problem,
        sample_observation(problem, jr.key(0), faint, window(problem)),
        window(problem),
    )
    assert 0.2 < float(image.std()) < 5.0


def test_snr_is_the_signal_norm_in_noise_units():
    # monte-carlo: snr == sqrt(2 * sum |clean|^2 / measured noise variance) over the window;
    # the 2 is the one in <x|y> = 4 Re sum x y^* / (S t_obs) that `power` halves away.
    # noise_psd is finite and positive everywhere the window reaches, so no mask is needed
    problem = LisaGB(n_sources=3, t_obs=1.0e6, f0_range=(3.0e-3, 3.2e-3))
    p = sample_physical(problem, jr.key(9), window(problem))
    clean = clean_signal(problem, p, window(problem))

    def accumulate(total, key):
        residual = sample_observation(problem, key, p, window(problem)) - clean
        return total + jnp.abs(residual) ** 2, None

    summed, _ = jax.lax.scan(
        accumulate,
        jnp.zeros_like(clean, dtype=jnp.float64),
        jr.split(jr.key(20), 4096),
    )
    var = summed / 4096
    assert jnp.allclose(
        snr(problem, p, window(problem)),
        jnp.sqrt(2.0 * jnp.sum(jnp.abs(clean) ** 2 / var)),
        rtol=0.03,
    )
