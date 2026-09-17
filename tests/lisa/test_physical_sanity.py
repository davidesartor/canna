"""Physical sanity: PSDs are positive, SNR is nonnegative, zero source is silent."""

import jax
import jax.numpy as jnp
import jax.random as jr

from canna.lisa import LisaGB
from ._helpers import window


def test_noise_psd_is_strictly_positive():
    problem = LisaGB()
    f = jnp.linspace(1e-4, 1e-2, 64)
    psd = problem.noise_psd(f)
    assert jnp.all(psd > 0)


def test_snr_is_nonnegative():
    problem = LisaGB(n_sources=2)
    p = problem.sample_physical(jr.key(40), window(problem))
    s = problem.snr(p, window(problem))
    assert s >= 0


def test_clean_signal_zero_for_all_zero_physical_params():
    # an all-zero row means zero amplitude among the 8 numbers
    # regardless of column layout, so the noiseless signal must vanish.
    problem = LisaGB(n_sources=1)
    p = jnp.zeros((1, 8))
    o = problem.clean_signal(p, window(problem))
    assert jnp.allclose(o, jnp.zeros_like(o), atol=1e-10)


def test_snr_zero_for_all_zero_physical_params():
    problem = LisaGB(n_sources=1)
    p = jnp.zeros((1, 8))
    s = problem.snr(p, window(problem))
    assert jnp.allclose(s, 0.0, atol=1e-8)


def test_clean_signal_frequency_axis_is_even_and_wdm_aligned():
    problem = LisaGB(n_sources=1)
    p = problem.sample_physical(jr.key(42), window(problem))
    o = problem.clean_signal(p, window(problem))
    n_freq = o.shape[-2]
    assert n_freq % 2 == 0
    block = problem.wdm_freq_bands * problem.patch_downsample
    assert (n_freq // 2) % block == 0


def test_clean_signal_frequency_axis_independent_of_sampling_step():
    # the F axis is sized in clean_signal from f0_range/t_obs/response_points/
    # wdm_freq_bands/patch_downsample; sampling_step is never referenced there.
    p_key = jr.key(43)
    fine = LisaGB(n_sources=1, sampling_step=1.0)
    coarse = LisaGB(n_sources=1, sampling_step=20.0)
    f = window(fine)
    p = fine.sample_physical(p_key, f)
    assert fine.clean_signal(p, f).shape == coarse.clean_signal(p, f).shape


def test_sample_physical_columns_respect_declared_ranges_and_angle_domains():
    # Product(f0, chirp_mass, amp, sky, orientation, phi0) fixes physical columns 0..7,
    # and Isotropic's chart.backward emits (azimuth in [0, 2pi), latitude in
    # [-pi/2, pi/2]) per 2-angle block.
    problem = LisaGB(n_sources=8)
    p = problem.sample_physical(jr.key(44), window(problem))
    f0, chirp_mass, amp = p[..., 0], p[..., 1], p[..., 2]
    sky_azimuth, sky_latitude = p[..., 3], p[..., 4]
    orient_azimuth, orient_latitude = p[..., 5], p[..., 6]
    phi0 = p[..., 7]
    mc_lo, mc_hi = problem.chirp_mass_range
    assert jnp.all((f0 >= problem.f0_range[0]) & (f0 <= problem.f0_range[1]))
    assert jnp.all((chirp_mass >= mc_lo) & (chirp_mass <= mc_hi))
    amp_lo, amp_hi = problem.a_window(window(problem))
    assert jnp.all((amp >= amp_lo) & (amp <= amp_hi))
    assert jnp.all((sky_azimuth >= 0) & (sky_azimuth < 2 * jnp.pi))
    assert jnp.all((sky_latitude >= -jnp.pi / 2) & (sky_latitude <= jnp.pi / 2))
    assert jnp.all((orient_azimuth >= 0) & (orient_azimuth < 2 * jnp.pi))
    assert jnp.all((orient_latitude >= -jnp.pi / 2) & (orient_latitude <= jnp.pi / 2))
    assert jnp.all((phi0 >= 0) & (phi0 < 2 * jnp.pi))


def test_clean_signal_scales_linearly_with_amplitude():
    # jaxgb's waveform model has "amp" (physical column 2) as an overall linear
    # prefactor; a missing constant factor in the hermitian-band construction would not
    # break linearity in amplitude, only its absolute normalisation.
    problem = LisaGB(n_sources=1)
    p = problem.sample_physical(jr.key(45), window(problem))
    scaled = p.at[..., 2].multiply(2.0)
    o = problem.clean_signal(p, window(problem))
    o_scaled = problem.clean_signal(scaled, window(problem))
    assert jnp.allclose(o_scaled, 2.0 * o, atol=1e-20, rtol=1e-6)


def test_sample_physical_supports_zero_sources():
    # n_sources=0 edge case: LisaGB's indexing stays shape-safe (get_kmin on a
    # length-0 slice, `.at[idxs].add` with empty idxs), and jaxgb.JaxGB.get_tdi/get_kmin
    # tolerate an empty source axis.
    problem = LisaGB(n_sources=0)
    p = problem.sample_physical(jr.key(47), window(problem))
    assert p.shape == (0, 8)
    o = problem.clean_signal(p, window(problem))
    assert jnp.allclose(o, jnp.zeros_like(o))


def test_sky_is_uniform_on_the_sphere():
    problem = LisaGB(n_sources=3)
    draws = jax.vmap(problem.sample_physical, in_axes=(0, None))(
        jr.split(jr.key(6), 4096), window(problem)
    )
    lon, lat = draws[..., 3], draws[..., 4]
    sky = jnp.stack(
        [jnp.cos(lat) * jnp.cos(lon), jnp.cos(lat) * jnp.sin(lon), jnp.sin(lat)], -1
    )
    assert jnp.allclose(jnp.mean(sky, axis=(0, 1)), 0.0, atol=0.03)


def test_orientation_is_uniform_on_the_sphere():
    problem = LisaGB(n_sources=3)
    draws = jax.vmap(problem.sample_physical, in_axes=(0, None))(
        jr.split(jr.key(7), 4096), window(problem)
    )
    psi, iota = draws[..., 5], draws[..., 6]
    orientation = jnp.stack(
        [jnp.cos(iota) * jnp.cos(psi), jnp.cos(iota) * jnp.sin(psi), jnp.sin(iota)], -1
    )
    assert jnp.allclose(jnp.mean(orientation, axis=(0, 1)), 0.0, atol=0.03)


def test_clean_signal_sums_over_its_sources():
    problem = LisaGB(
        n_sources=3, t_obs=1.0e6, wdm_freq_bands=128, f0_range=(3.0e-3, 3.2e-3)
    )
    p = problem.sample_physical(jr.key(12), window(problem))
    total = problem.clean_signal(p, window(problem))
    parts = sum(
        problem.clean_signal(p[i : i + 1], window(problem))
        for i in range(problem.n_sources)
    )
    assert jnp.allclose(total, parts, rtol=1e-6, atol=0.0)


def test_noise_psd_is_even_in_frequency():
    problem = LisaGB()
    freqs = jnp.linspace(1e-4, 0.1, 512)
    assert jnp.allclose(problem.noise_psd(freqs), problem.noise_psd(-freqs))


def test_null_channel_differs_from_the_science_channels():
    problem = LisaGB()
    psd = problem.noise_psd(jnp.linspace(1e-4, 1e-2, 64))
    a, t = psd[..., 0], psd[..., 2]
    assert not jnp.allclose(a / a.max(), t / t.max())
