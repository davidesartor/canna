"""Sources are an unordered set: reordering rows must not change physics."""

import jax.numpy as jnp
import jax.random as jr

from canna.lisa import LisaGB
from ._helpers import window



def test_clean_signal_invariant_to_source_permutation():
    problem = LisaGB(n_sources=3)
    p = problem.sample_physical(jr.key(20), window(problem))
    perm = jnp.array([2, 0, 1])
    o = problem.clean_signal(p, window(problem))
    o_perm = problem.clean_signal(p[perm], window(problem))
    # strains are O(1e-18), so the tolerance has to be relative to the draw's own scale
    assert jnp.allclose(o, o_perm, atol=1e-12 * float(jnp.abs(o).max()), rtol=1e-6)


def test_snr_invariant_to_source_permutation():
    problem = LisaGB(n_sources=3)
    p = problem.sample_physical(jr.key(21), window(problem))
    perm = jnp.array([2, 0, 1])
    s = problem.snr(p, window(problem))
    s_perm = problem.snr(p[perm], window(problem))
    assert jnp.allclose(s, s_perm, atol=1e-6, rtol=1e-6)


def test_sample_observation_invariant_to_source_permutation():
    # the noise draw depends only on key and shape, so the whole noisy observation
    # inherits clean_signal's permutation invariance
    problem = LisaGB(n_sources=3)
    p = problem.sample_physical(jr.key(22), window(problem))
    perm = jnp.array([2, 0, 1])
    key = jr.key(23)
    o = problem.sample_observation(key, p, window(problem))
    o_perm = problem.sample_observation(key, p[perm], window(problem))
    assert jnp.allclose(o, o_perm, atol=1e-12 * float(jnp.abs(o).max()), rtol=1e-6)


def test_preprocess_invariant_to_source_permutation():
    problem = LisaGB(n_sources=3)
    p = problem.sample_physical(jr.key(24), window(problem))
    perm = jnp.array([2, 0, 1])
    img = problem.preprocess(problem.clean_signal(p, window(problem)), window(problem))
    img_perm = problem.preprocess(problem.clean_signal(p[perm], window(problem)), window(problem))
    assert jnp.allclose(img, img_perm, atol=1e-6, rtol=1e-6)


def test_iota_change_on_one_source_leaves_a_siblings_band_untouched():
    # the p[...,6] iota colatitude flip must not leak across summed per-source segments.
    # the short baseline is what lets one window hold two sources a decade apart in f0
    problem = LisaGB(n_sources=2, t_obs=5.0e5, f0_range=(1.0e-3, 6.0e-3))
    p = jnp.array(
        [
            [1e-3, 0.3, 4e-23, 0.3, 0.2, 0.7, 0.2, 1.1],
            [6e-3, 0.5, 6e-23, 0.3, 0.2, 0.7, 0.2, 1.1],
        ]
    )
    p_pert = p.at[1, 6].set(-0.4)
    o = problem.clean_signal(p, window(problem))
    o_pert = problem.clean_signal(p_pert, window(problem))
    kmin0 = int(problem.response.get_kmin(p[:1, 0])[0] - problem.window_start(window(problem)))
    band0 = slice(kmin0, kmin0 + problem.response_points)
    assert float(jnp.abs(o[band0]).max()) > 0.0
    assert jnp.allclose(o[band0], o_pert[band0], atol=1e-30)
