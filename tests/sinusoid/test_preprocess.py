"""NoisySinusoid.preprocess: the band-crop/re-centering onto the WDM grid, including
the mmin>0 (recentering active) regime and the wide-band overflow limitation."""

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from canna.sinusoid import NoisySinusoid


@pytest.fixture
def recentered():
    # freq_range=(0.1,0.4) at this config gives mmin=1, so the kept block shifts off kmin=0
    return NoisySinusoid(
        n_sources=1,
        t_obs=256.0,
        sampling_step=1.0,
        wdm_freq_bands=6,
        patch_downsample=4,
        freq_range=(0.1, 0.4),
    )


def test_preprocess_raises_on_a_band_too_wide_for_the_grid():
    """A band wide relative to wdm_freq_bands*patch_downsample overruns the WDM grid; the
    error surfaces (unguarded) from wdm_transform, not from a NoisySinusoid-layer check.
    """
    problem = NoisySinusoid(
        n_sources=1,
        t_obs=256.0,
        wdm_freq_bands=6,
        patch_downsample=4,
        freq_range=(0.02, 0.49),
    )
    with pytest.raises(Exception):
        problem.preprocess(jnp.zeros((256, 2)))


def test_preprocess_succeeds_on_a_narrower_band():
    problem = NoisySinusoid(
        n_sources=1,
        t_obs=256.0,
        wdm_freq_bands=6,
        patch_downsample=4,
        freq_range=(0.02, 0.3),
    )
    assert problem.preprocess(jnp.zeros((256, 2))).shape[-2] == 6


def test_preprocess_batched_matches_looped_when_recentered(recentered):
    keys = jr.split(jr.key(7), 3)
    p_batch = jnp.stack([recentered.sample_physical(k) for k in keys])
    o_batch = jax.vmap(recentered.sample_observation)(keys, p_batch)
    batched = recentered.preprocess(o_batch)
    looped = jnp.stack([recentered.preprocess(o_batch[i]) for i in range(3)])
    assert jnp.allclose(batched, looped, atol=1e-6)


def test_preprocess_peak_channel_tracks_frequency_when_recentered(recentered):
    low, high = recentered.freq_range
    freqs = jnp.exp(jnp.linspace(jnp.log(low), jnp.log(high), 5))
    peaks = [
        int(
            jnp.argmax(
                jnp.abs(
                    recentered.preprocess(
                        recentered.clean_signal(jnp.array([[1.0, float(f), 0.0]]))
                    )
                ).sum(axis=(0, 2))
            )
        )
        for f in freqs
    ]
    assert all(lo <= hi for lo, hi in zip(peaks[:-1], peaks[1:]))


def test_preprocess_finite_when_recentered(recentered):
    assert jnp.all(jnp.isfinite(recentered.preprocess(jnp.zeros((256, 2)))))


def test_preprocess_finite_across_sampling_step_at_fixed_length():
    unit_step = NoisySinusoid(
        n_sources=1,
        t_obs=256.0,
        sampling_step=1.0,
        wdm_freq_bands=6,
        patch_downsample=4,
        freq_range=(0.02, 0.3),
    )
    half_step = NoisySinusoid(
        n_sources=1,
        t_obs=128.0,
        sampling_step=0.5,
        wdm_freq_bands=6,
        patch_downsample=4,
        freq_range=(0.02, 0.3),
    )
    o = jnp.zeros((256, 2))
    assert jnp.all(jnp.isfinite(unit_step.preprocess(o)))
    assert jnp.all(jnp.isfinite(half_step.preprocess(o)))


def test_preprocess_finite_on_extreme_input_when_recentered(recentered):
    assert jnp.all(jnp.isfinite(recentered.preprocess(jnp.full((256, 2), 1e30))))


def test_preprocess_shape_small_config():
    """Small config (t_obs=256, wdm_freq_bands=8): f==8 and nt==8."""
    problem = NoisySinusoid(
        n_sources=2,
        t_obs=256.0,
        sampling_step=1.0,
        wdm_freq_bands=8,
        patch_downsample=4,
    )
    key = jr.key(0)
    p = problem.sample_physical(key)
    o = problem.sample_observation(key, p)
    img = problem.preprocess(o)
    assert img.shape[-3:] == (8, 8, 2)


def test_preprocess_freq_axis_equals_wdm_freq_bands_wide_band():
    """A wide band (freq_range=(0.02,0.49)) keeps the f axis at wdm_freq_bands; only nt grows."""
    problem = NoisySinusoid(
        n_sources=2,
        t_obs=256.0,
        sampling_step=1.0,
        wdm_freq_bands=8,
        patch_downsample=4,
        freq_range=(0.02, 0.49),
    )
    key = jr.key(0)
    p = problem.sample_physical(key)
    o = problem.sample_observation(key, p)
    img = problem.preprocess(o)
    assert img.shape[-2] == problem.wdm_freq_bands


@pytest.mark.parametrize(
    "t_obs,wdm_freq_bands,patch_downsample,freq_range",
    [
        (256.0, 8, 4, (0.01, 0.1)),
        (512.0, 16, 4, (0.02, 0.2)),
        (1024.0, 10, 2, (0.03, 0.3)),
    ],
)
def test_preprocess_output_dims_are_patch_divisible(
    t_obs, wdm_freq_bands, patch_downsample, freq_range
):
    """Both height (nt) and freq fold into patch_downsample-square patches, regardless
    of the wdm_freq_bands multiplier."""
    problem = NoisySinusoid(
        n_sources=2,
        t_obs=t_obs,
        sampling_step=1.0,
        wdm_freq_bands=wdm_freq_bands,
        patch_downsample=patch_downsample,
        freq_range=freq_range,
    )
    key = jr.key(0)
    p = problem.sample_physical(key)
    o = problem.sample_observation(key, p)
    img = problem.preprocess(o)
    assert img.shape[-3] % patch_downsample == 0
    assert img.shape[-2] % patch_downsample == 0


def test_preprocess_channel_swap_permutes_output_channels():
    """to_wdm vmaps over the last (channel) axis independently (in_axes=-1, out_axes=-1),
    so swapping the sin/cos input channels exactly swaps the two output channels."""
    problem = NoisySinusoid(
        n_sources=2,
        t_obs=256.0,
        sampling_step=1.0,
        wdm_freq_bands=8,
        patch_downsample=4,
    )
    key = jr.key(0)
    p = problem.sample_physical(key)
    o = problem.sample_observation(key, p)
    o_swapped = o[..., ::-1]
    img = problem.preprocess(o)
    img_swapped = problem.preprocess(o_swapped)
    assert jnp.allclose(img_swapped, img[..., ::-1], atol=1e-6)


def test_preprocess_batched_matches_looped_per_sample():
    """jnp.vectorize's batched call over a leading dim matches looping preprocess
    per-sample and stacking."""
    problem = NoisySinusoid(
        n_sources=2,
        t_obs=256.0,
        sampling_step=1.0,
        wdm_freq_bands=8,
        patch_downsample=4,
    )
    keys = jr.split(jr.key(0), 3)
    p_batch = jnp.stack([problem.sample_physical(k) for k in keys])
    o_batch = jax.vmap(lambda k, p: problem.sample_observation(k, p))(keys, p_batch)
    batched = problem.preprocess(o_batch)
    looped = jnp.stack([problem.preprocess(o_batch[i]) for i in range(3)])
    assert jnp.allclose(batched, looped, atol=1e-6)
