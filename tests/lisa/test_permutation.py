"""Reordering sources must not change the physics; sampling lists them loudest first."""

from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from canna.lisa.problem import LisaGB, f0_window, window_start
from canna.lisa.simulator import (
    clean_signal,
    preprocess,
    sample_observation,
    sample_physical,
    snr,
    source_snrs,
)
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
            [1e-3, 0.3, 4e-23, 0.3, 0.2, 0.7, 0.2, 1.1],
            [6e-3, 0.5, 6e-23, 0.3, 0.2, 0.7, 0.2, 1.1],
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


def test_sample_physical_lists_the_sources_loudest_first():
    problem = LisaGB(n_sources=4)
    f = window(problem)
    draw = eqx.filter_jit(jax.vmap(partial(sample_physical, problem), in_axes=(0, None)))
    p = draw(jr.split(jr.key(30), 64), f)
    snrs = source_snrs(problem, p, f)
    assert snrs.shape == (64, 4)
    assert jnp.all(jnp.diff(snrs, axis=-1) <= 0)


def test_loudest_first_is_a_reordering_of_the_draw():
    problem = LisaGB(n_sources=4)
    f = window(problem)
    key = jr.key(31)
    p = sample_physical(problem, key, f)
    # the unsorted draw's f0 and target snrs, off sample_physical's own keys
    keys = jr.split(key, 8)
    log_f = jnp.log(jnp.stack(f0_window(problem, f)))
    f0 = jnp.exp(jr.uniform(keys[0], (4, 1), minval=log_f[0], maxval=log_f[1]))[:, 0]
    log_snr = jnp.log(jnp.asarray(problem.snr_range))
    target = jnp.exp(jr.uniform(keys[2], (4,), minval=log_snr[0], maxval=log_snr[1]))
    order = jnp.argsort(-target)
    # every row moves whole: its f0 and its own snr follow the same permutation
    assert jnp.allclose(p[:, 0], f0[order], rtol=1e-12, atol=0.0)
    assert jnp.allclose(source_snrs(problem, p, f), target[order], rtol=1e-6)
