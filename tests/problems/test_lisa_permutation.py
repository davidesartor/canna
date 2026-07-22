"""Sources are an unordered set: reordering rows must not change physics."""

import jax.numpy as jnp
import jax.random as jr

from canna.problems import LisaGB


def test_clean_signal_invariant_to_source_permutation():
    problem = LisaGB(n_sources=3)
    p = problem.sample_physical(jr.key(20))
    perm = jnp.array([2, 0, 1])
    o = problem.clean_signal(p)
    o_perm = problem.clean_signal(p[perm])
    assert jnp.allclose(o, o_perm, atol=1e-6, rtol=1e-6)


def test_snr_invariant_to_source_permutation():
    problem = LisaGB(n_sources=3)
    p = problem.sample_physical(jr.key(21))
    perm = jnp.array([2, 0, 1])
    s = problem.snr(p)
    s_perm = problem.snr(p[perm])
    assert jnp.allclose(s, s_perm, atol=1e-6, rtol=1e-6)


def test_sample_observation_clean_invariant_to_source_permutation():
    problem = LisaGB(n_sources=3)
    p = problem.sample_physical(jr.key(22))
    perm = jnp.array([2, 0, 1])
    key = jr.key(23)
    o = problem.sample_observation(key, p, clean=True)
    o_perm = problem.sample_observation(key, p[perm], clean=True)
    assert jnp.allclose(o, o_perm, atol=1e-6, rtol=1e-6)


def test_preprocess_invariant_to_source_permutation():
    problem = LisaGB(n_sources=3)
    p = problem.sample_physical(jr.key(24))
    perm = jnp.array([2, 0, 1])
    img = problem.preprocess(problem.clean_signal(p))
    img_perm = problem.preprocess(problem.clean_signal(p[perm]))
    assert jnp.allclose(img, img_perm, atol=1e-6, rtol=1e-6)


def test_iota_change_on_one_source_leaves_a_siblings_band_untouched():
    # the p[...,6] iota colatitude flip must not leak across summed per-source segments
    problem = LisaGB(n_sources=2)
    p = jnp.array(
        [
            [1e-3, 3e-17, 4e-23, 0.3, 0.2, 0.7, 0.2, 1.1],
            [6e-3, 5e-17, 6e-23, 0.3, 0.2, 0.7, 0.2, 1.1],
        ]
    )
    p_pert = p.at[1, 6].set(-0.4)
    o = problem.clean_signal(p)
    o_pert = problem.clean_signal(p_pert)
    fmin, _ = problem.f0_range_bins
    kmin0 = int(problem.response.get_kmin(p[:1, 0])[0]) - fmin
    band0 = slice(kmin0, kmin0 + problem.response_points)
    assert jnp.allclose(o[band0], o_pert[band0], atol=1e-30)
