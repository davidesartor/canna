"""clean_signal's F-axis sizing arithmetic and the hermitian build under
sampling_step=0.25, wdm_freq_bands=4096, patch_downsample=8."""

import jax.numpy as jnp
import jax.random as jr

from canna.problems import LisaGB


def test_clean_signal_f_axis_identical_across_short_baselines_below_overshoot_floor():
    # n = (n_raw // k + 1) * k with k = wdm_freq_bands * patch_downsample (now 32768).
    # n_raw = int(f0_range[1] * t_obs + response_points / 2) stays far below k for any
    # t_obs from a day to a week (12e-3 Hz * 7 days + 128 ~= 7385 << 32768), so n --
    # and hence clean_signal's whole F axis -- floors to the same constant regardless
    # of which of these two very different observation lengths is used.
    p_key = jr.key(71)
    day = LisaGB(n_sources=1, t_obs=86400.0)
    week = LisaGB(n_sources=1, t_obs=7 * 86400.0)
    p = day.sample_physical(p_key)
    assert day.clean_signal(p).shape == week.clean_signal(p).shape


def test_clean_signal_f_axis_floors_at_2k_for_short_t_obs():
    # same overshoot: for t_obs = 1 day, n_raw ~= 1164 << k = 32768, so n rounds *up*
    # all the way to k itself (not to the nearest small multiple of anything close to
    # n_raw), and F = 2 * n = 2 * k.
    problem = LisaGB(n_sources=1, t_obs=86400.0)
    p = problem.sample_physical(jr.key(70))
    o = problem.clean_signal(p)
    k = problem.wdm_freq_bands * problem.patch_downsample
    assert o.shape[-2] == 2 * k


def test_wdm_shape_invariants_hold_across_a_range_of_observation_baselines():
    # the two invariants clean_signal/preprocess must jointly maintain (F divisible by
    # 2*wdm_freq_bands*patch_downsample, and the resulting image time axis divisible
    # by patch_downsample) should hold whether t_obs undershoots the overshoot floor
    # (1 day), sits at the shipped 30-day baseline, or is a multi-year baseline.
    p_key = jr.key(69)
    for t_obs in (
        86400.0,
        2592000.0,
        2 * 365.25 * 24 * 60 * 60,
        10 * 365 * 24 * 60 * 60,
    ):
        problem = LisaGB(n_sources=1, t_obs=t_obs)
        p = problem.sample_physical(p_key)
        o = problem.clean_signal(p)
        img = problem.preprocess(o)
        assert (
            o.shape[-2] % (2 * problem.wdm_freq_bands * problem.patch_downsample) == 0
        )
        assert img.shape[-3] % problem.patch_downsample == 0


def test_conditioning_image_shape_at_30day_baseline_config():
    # regression pin for the shipped 30-day mock config (LisaGB-M.yaml: t_obs=2592000.0,
    # wdm_freq_bands=4096, patch_downsample=16): F floors to 2*k=131072, so the time axis
    # is 32, and the trailing wdm[..., 1:] drop leaves frequency axis == 4096.
    problem = LisaGB(
        n_sources=1, t_obs=2592000.0, wdm_freq_bands=4096, patch_downsample=16
    )
    p = problem.sample_physical(jr.key(60))
    o = problem.clean_signal(p)
    img = problem.preprocess(o)
    assert img.shape[-3] == 32
    assert img.shape[-2] == 4096


def test_conditioning_frequency_axis_is_divisible_by_mmdit_b_patch_stages():
    # MMDiT-B.yaml sets patch_stages=4, so Patchify's four Fold2d halvings need both
    # image axes divisible by 2**4=16. On the shipped 30-day config the frequency axis
    # is wdm_freq_bands (4096) and the time axis is 32 -- both clear multiples of 16.
    problem = LisaGB(
        n_sources=1, t_obs=2592000.0, wdm_freq_bands=4096, patch_downsample=16
    )
    p = problem.sample_physical(jr.key(61))
    o = problem.clean_signal(p)
    img = problem.preprocess(o)
    assert img.shape[-2] % 16 == 0
    assert img.shape[-3] % 16 == 0


def test_clean_signal_is_hermitian_symmetric():
    # clean_signal's own comment claims "the band is hermitian by build": a real time
    # series has X[-f] == conj(X[f]). Check that end to end, independent of jaxgb's
    # internals, using the fftfreq-style index convention (index j <-> bin -j is index
    # (-j) mod F).
    problem = LisaGB(n_sources=1)
    p = problem.sample_physical(jr.key(72))
    o = problem.clean_signal(p)
    mirrored = jnp.conj(jnp.roll(jnp.flip(o, axis=-2), 1, axis=-2))
    assert jnp.allclose(o, mirrored, atol=1e-15)
