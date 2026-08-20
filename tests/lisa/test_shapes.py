"""Shape contracts for LisaGB's public surface."""

from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from canna.lisa import LisaGB, train_sample
from ._helpers import window


# a narrow band-limited config keeps the rank-2 batch broadcasts cheap
SMALL = dict(
    n_sources=2,
    t_obs=1.0e6,
    sampling_step=0.25,
    wdm_freq_bands=64,
    patch_downsample=4,
    f0_range=(3.0e-3, 3.2e-3),
)


def test_sample_physical_shape_default_n_sources():
    problem = LisaGB()
    p = problem.sample_physical(jr.key(0), window(problem))
    assert p.shape == (problem.n_sources, 8)


def test_sample_physical_shape_multi_source():
    problem = LisaGB(n_sources=5)
    p = problem.sample_physical(jr.key(0), window(problem))
    assert p.shape == (5, 8)


def test_sample_flow_shape_is_eleven_coords():
    problem = LisaGB(n_sources=3)
    x = problem.sample_flow(jr.key(0), window(problem))
    assert x.shape == (3, 11)


def test_noise_psd_scalar_frequency_gives_three_channels():
    problem = LisaGB()
    psd = problem.noise_psd(jnp.array(1e-3))
    assert psd.shape == (3,)


def test_noise_psd_vector_frequency_preserves_leading_axis():
    problem = LisaGB()
    f = jnp.linspace(1e-4, 1e-2, 128)
    psd = problem.noise_psd(f)
    assert psd.shape == (128, 3)


def test_noise_psd_batched_frequency_preserves_all_leading_axes():
    problem = LisaGB()
    f = jnp.ones((4, 128)) * 1e-3
    psd = problem.noise_psd(f)
    assert psd.shape == (4, 128, 3)


def test_clean_signal_output_has_three_channels():
    problem = LisaGB(n_sources=2)
    p = problem.sample_physical(jr.key(0), window(problem))
    o = problem.clean_signal(p, window(problem))
    assert o.shape[-1] == 3
    assert jnp.iscomplexobj(o)


def test_clean_signal_preserves_leading_batch_axis():
    problem = LisaGB(n_sources=2)
    p = jax.vmap(problem.sample_physical, in_axes=(0, None))(
        jr.split(jr.key(0), 4), window(problem)
    )
    o = problem.clean_signal(p, window(problem))
    assert o.shape[0] == 4
    assert o.shape[-1] == 3


def test_snr_scalar_for_unbatched_input():
    problem = LisaGB(n_sources=2)
    p = problem.sample_physical(jr.key(0), window(problem))
    s = problem.snr(p, window(problem))
    assert s.shape == ()


def test_snr_preserves_leading_batch_axis():
    problem = LisaGB(n_sources=2)
    p = jax.vmap(problem.sample_physical, in_axes=(0, None))(
        jr.split(jr.key(0), 4), window(problem)
    )
    s = problem.snr(p, window(problem))
    assert s.shape == (4,)


def test_preprocess_output_has_three_channels():
    problem = LisaGB(n_sources=1)
    p = problem.sample_physical(jr.key(0), window(problem))
    o = problem.clean_signal(p, window(problem))
    img = problem.preprocess(o, window(problem))
    assert img.shape[-1] == 3
    assert img.ndim == o.ndim + 1


def test_preprocess_preserves_leading_batch_axis():
    problem = LisaGB(n_sources=1)
    p = jax.vmap(problem.sample_physical, in_axes=(0, None))(
        jr.split(jr.key(0), 4), window(problem)
    )
    o = problem.clean_signal(p, window(problem))
    img = problem.preprocess(o, window(problem))
    assert img.shape[0] == 4
    assert img.shape[-1] == 3


def test_preprocess_time_axis_divisible_by_patch_downsample():
    # the window is wdm_freq_bands * wdm_times // 2 bins wide, so the image time axis is
    # wdm_times, which __post_init__ forces to be a multiple of patch_downsample
    problem = LisaGB(n_sources=1)
    p = problem.sample_physical(jr.key(48), window(problem))
    o = problem.clean_signal(p, window(problem))
    img = problem.preprocess(o, window(problem))
    assert img.shape[-3] % problem.patch_downsample == 0


def test_train_sample_shapes_line_up():
    problem = LisaGB(n_sources=3)
    width = 11
    sample = train_sample(problem, jr.key(50))
    assert sample.xt.shape == (3, width)
    assert sample.dx.shape == (3, width)
    assert sample.t.shape == ()
    assert sample.x_target.shape == (3, width)


def test_train_sample_is_finite():
    problem = LisaGB(n_sources=2)
    sample = eqx.filter_jit(partial(train_sample, problem))(jr.key(51))
    assert jnp.all(jnp.isfinite(sample.xt))
    assert jnp.all(jnp.isfinite(sample.dx))
    assert jnp.isfinite(sample.t)


def test_train_sample_same_key_gives_same_sample():
    problem = LisaGB(n_sources=2)
    a = train_sample(problem, jr.key(52))
    b = train_sample(problem, jr.key(52))
    assert jnp.allclose(a.xt, b.xt)
    assert jnp.allclose(a.dx, b.dx)
    assert jnp.allclose(a.t, b.t)


def test_preprocess_frequency_axis_is_wdm_freq_bands():
    problem = LisaGB(n_sources=1)
    p = problem.sample_physical(jr.key(49), window(problem))
    o = problem.clean_signal(p, window(problem))
    img = problem.preprocess(o, window(problem))
    assert img.shape[-2] == problem.wdm_freq_bands


def test_clean_signal_broadcasts_over_two_leading_batch_axes():
    # clean_signal is a jnp.vectorize ufunc, so it must broadcast over any number of
    # leading batch axes, not just the single vmap axis.
    problem = LisaGB(**SMALL)
    ps = jax.vmap(problem.sample_physical, in_axes=(0, None))(
        jr.split(jr.key(0), 6), window(problem)
    )
    ps_2d = ps.reshape(2, 3, *ps.shape[1:])
    h_2d = problem.clean_signal(ps_2d, window(problem))
    h_flat = problem.clean_signal(ps, window(problem))
    assert h_2d.shape == (2, 3, *h_flat.shape[1:])
    assert jnp.allclose(h_2d.reshape(h_flat.shape), h_flat, rtol=1e-6)


def test_preprocess_broadcasts_over_two_leading_batch_axes():
    problem = LisaGB(**SMALL)
    keys = jr.split(jr.key(0), 6)
    ps = jax.vmap(problem.sample_physical, in_axes=(0, None))(keys, window(problem))
    os_ = jax.vmap(lambda k, p: problem.sample_observation(k, p, window(problem)))(keys, ps)
    os_2d = os_.reshape(2, 3, *os_.shape[1:])
    img_2d = problem.preprocess(os_2d, window(problem))
    img_flat = problem.preprocess(os_, window(problem))
    assert img_2d.shape == (2, 3, *img_flat.shape[1:])
    assert jnp.allclose(img_2d.reshape(img_flat.shape), img_flat, atol=1e-5)


def test_snr_broadcasts_over_two_leading_batch_axes():
    problem = LisaGB(**SMALL)
    ps = jax.vmap(problem.sample_physical, in_axes=(0, None))(
        jr.split(jr.key(0), 6), window(problem)
    )
    ps_2d = ps.reshape(2, 3, *ps.shape[1:])
    snr_2d = problem.snr(ps_2d, window(problem))
    snr_flat = problem.snr(ps, window(problem))
    assert snr_2d.shape == (2, 3)
    assert jnp.allclose(snr_2d.reshape(-1), snr_flat, rtol=1e-5)
