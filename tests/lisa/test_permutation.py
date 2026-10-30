"""Reordering sources must not change the physics; training puts the sources in f0 order."""

import jax.numpy as jnp
import jax.random as jr

from canna.lisa.injections import sort_by_snr
from canna.lisa.problem import LisaGB, window_start
from canna.lisa.simulator import (
    clean_signal,
    preprocess,
    sample_observation,
    sample_physical,
    snr,
    source_snrs,
)
from canna.lisa.train import train_sample

from ._helpers import window


def test_clean_signal_invariant_to_source_permutation():
    problem = LisaGB(n_sources=3)
    p = sample_physical(problem, jr.key(20), window(problem))
    perm = jnp.array([2, 0, 1])
    o = clean_signal(problem, p, window(problem))
    o_perm = clean_signal(problem, p[perm], window(problem))
    # strains are O(1e-18), so the tolerance has to be relative to the draw's own scale
    assert jnp.allclose(o, o_perm, atol=1e-12 * float(jnp.abs(o).max()), rtol=1e-6)


def test_snr_invariant_to_source_permutation():
    problem = LisaGB(n_sources=3)
    p = sample_physical(problem, jr.key(21), window(problem))
    perm = jnp.array([2, 0, 1])
    s = snr(problem, p, window(problem))
    s_perm = snr(problem, p[perm], window(problem))
    assert jnp.allclose(s, s_perm, atol=1e-6, rtol=1e-6)


def test_sample_observation_invariant_to_source_permutation():
    # the noise draw depends only on key and shape, so the whole noisy observation
    # inherits clean_signal's permutation invariance
    problem = LisaGB(n_sources=3)
    p = sample_physical(problem, jr.key(22), window(problem))
    perm = jnp.array([2, 0, 1])
    key = jr.key(23)
    o = sample_observation(problem, key, p, window(problem))
    o_perm = sample_observation(problem, key, p[perm], window(problem))
    assert jnp.allclose(o, o_perm, atol=1e-12 * float(jnp.abs(o).max()), rtol=1e-6)


def test_preprocess_invariant_to_source_permutation():
    problem = LisaGB(n_sources=3)
    f = window(problem)
    p = sample_physical(problem, jr.key(24), f)
    perm = jnp.array([2, 0, 1])
    img = preprocess(problem, clean_signal(problem, p, f), f)
    img_perm = preprocess(problem, clean_signal(problem, p[perm], f), f)
    assert jnp.allclose(img, img_perm, atol=1e-6, rtol=1e-6)


def test_iota_change_on_one_source_leaves_a_siblings_band_untouched():
    # the p[...,6] iota colatitude flip must not leak across summed per-source segments.
    # the short baseline is what lets one window hold two sources a decade apart in f0
    problem = LisaGB(n_sources=2, t_obs=5.0e5, f0_range=(1.0e-3, 6.0e-3))
    p = jnp.array(
        [
            [1e-3, 1e-18, 4e-23, 0.3, 0.2, 0.7, 0.2, 1.1],
            [6e-3, 1e-16, 6e-23, 0.3, 0.2, 0.7, 0.2, 1.1],
        ]
    )
    p_pert = p.at[1, 6].set(-0.4)
    o = clean_signal(problem, p, window(problem))
    o_pert = clean_signal(problem, p_pert, window(problem))
    kmin0 = int(
        problem.response.get_kmin(p[:1, 0])[0] - window_start(problem, window(problem))
    )
    band0 = slice(kmin0, kmin0 + problem.response_points)
    assert float(jnp.abs(o[band0]).max()) > 0.0
    assert jnp.allclose(o[band0], o_pert[band0], atol=1e-30)


def test_sort_by_snr_reorders_whole_rows_loudest_first():
    problem = LisaGB(n_sources=4)
    f = window(problem)
    p = sample_physical(problem, jr.key(31), f)
    shown = sort_by_snr(problem, p, f)
    assert sorted(map(tuple, shown.tolist())) == sorted(map(tuple, p.tolist()))
    assert jnp.all(jnp.diff(source_snrs(problem, shown, f)) <= 0)


def test_train_sample_puts_the_sources_in_f0_order():
    """The frame's slots are f0-ordered, so x1's rows must be too."""
    problem = LisaGB(n_sources=4)
    for seed in range(4):
        batch = train_sample(problem, jr.key(42 + seed))
        assert jnp.all(jnp.diff(batch.x1[:, 0]) >= 0)
