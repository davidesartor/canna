"""Whitening round trip and the flat per-source chart: the sky as galactic (l, b) whitened by the prior, straight paths."""

from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from canna.lisa.chart import (
    SKY,
    flow_to_physical,
    physical_to_flow,
    sample_flow,
)
from canna.lisa.problem import LisaGB
from canna.lisa.simulator import sample_physical
from canna.lisa.train import train_sample
from ._helpers import window


def test_chart_roundtrip_recovers_physical_parameters():
    problem = LisaGB(n_sources=2)
    p = sample_physical(problem, jr.key(1), window(problem))
    x = physical_to_flow(problem, p, window(problem))
    p_back = flow_to_physical(problem, x, window(problem))
    # (psi, phi0) come back as some copy under the likelihood's symmetries, so they are checked through the chart
    others = jnp.asarray([0, 1, 2, 3, 4, 6])
    assert jnp.allclose(p_back[:, others], p[:, others], atol=1e-4, rtol=1e-4)
    assert jnp.allclose(
        physical_to_flow(problem, p_back, window(problem)), x, atol=1e-6
    )


def test_physical_to_flow_maps_eight_to_eight_per_source():
    problem = LisaGB(n_sources=2)
    p = sample_physical(problem, jr.key(2), window(problem))
    assert p.shape == (2, 8) and physical_to_flow(
        problem, p, window(problem)
    ).shape == (2, 8)


def test_physical_to_flow_equivariant_to_source_permutation():
    problem = LisaGB(n_sources=3)
    p = sample_physical(problem, jr.key(3), window(problem))
    perm = jnp.array([2, 0, 1])
    x = physical_to_flow(problem, p, window(problem))
    x_of_permuted = physical_to_flow(problem, p[perm], window(problem))
    assert jnp.allclose(x_of_permuted, x[perm], atol=1e-6, rtol=1e-6)


def test_flow_draws_whiten_the_sky():
    # the sky is drawn and whitened independently of the band, so a narrow one will do
    problem = LisaGB(n_sources=2, t_obs=1.0e6, f0_range=(3.0e-3, 3.2e-3))
    f = window(problem)
    x = jax.lax.map(
        lambda key: sample_flow(problem, key, f),
        jr.split(jr.key(0), 8192),
        batch_size=512,
    )[..., SKY].reshape(-1, 2)
    assert jnp.allclose(jnp.mean(x, axis=0), 0.0, atol=0.1)
    assert jnp.allclose(jnp.std(x, axis=0), 1.0, atol=0.1)


def test_train_sample_walks_a_straight_line_and_is_finite():
    problem = LisaGB(n_sources=2)
    sample = eqx.filter_jit(partial(train_sample, problem, time_power=3.0))(jr.key(0))
    assert all(bool(jnp.all(jnp.isfinite(a))) for a in sample)
    remaining = ((1 - sample.t) ** 3.0)[:, None, None]
    assert jnp.allclose(sample.xt + remaining * sample.dx, sample.x_target, atol=1e-6)


def test_sample_flow_is_the_charted_prior_draw_of_the_same_key():
    problem = LisaGB(n_sources=2)
    f = window(problem)
    key = jr.key(13)
    x = sample_flow(problem, key, f)
    p = sample_physical(problem, key, f)
    assert jnp.allclose(x, physical_to_flow(problem, p, f), rtol=1e-10, atol=1e-10)
