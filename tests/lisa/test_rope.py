"""RoPE in absolute frequency: the sources' bins, the angles, the rotation, and the network with it."""

from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from canna.lisa import config
from canna.lisa.chart import flow_to_physical, sample_flow
from canna.lisa.network import LisaFlow, reference_frame
from canna.lisa.problem import LisaGB, window_start
from canna.lisa.train import train_sample
from canna.networks.mmdit import rotate
from ._helpers import window


def problem(name="narrow-B"):
    return LisaGB(**config.load(name).problem)


def test_source_bins_agree_with_flow_to_physical():
    for name, atol in (("narrow-B", 1e-3), ("full-B", 2e-2)):
        gb = problem(name)
        f = window(gb)
        x = sample_flow(gb, jr.key(1), f)
        f0_bins = flow_to_physical(gb, x, f)[:, 0] * gb.t_obs - window_start(gb, f)
        net = LisaFlow(gb, 32, 2, 1, key=jr.key(0))
        np.testing.assert_allclose(
            net.source_bins(x[:, 0], f), f0_bins, atol=atol, rtol=0
        )


def test_rope_angles_turn_with_the_absolute_frequency():
    """From 0 Hz, not the window: the same frequency has the same angle in every window."""
    net = LisaFlow(problem(), 32, 2, 1, key=jr.key(0))
    freq = jnp.asarray([0.0, 1.0e-3, 3.2e-3])
    turns = 2 * np.pi * np.asarray(freq)[:, None] / np.asarray(net.rope_periods)
    angles = net.rope_angles(freq)
    assert angles.dtype == jnp.float32
    np.testing.assert_allclose(jnp.cos(angles), np.cos(turns), atol=1e-5)
    np.testing.assert_allclose(jnp.sin(angles), np.sin(turns), atol=1e-5)


def test_rotation_makes_q_dot_k_depend_on_the_position_difference_only():
    q, k = jr.normal(jr.key(0), (1, 1, 8)), jr.normal(jr.key(1), (1, 1, 8))
    periods = jnp.array([2.0, 5.0, 17.0, 64.0])

    def score(pq, pk):
        aq, ak = 2 * jnp.pi * pq / periods, 2 * jnp.pi * pk / periods
        rq = rotate(q, jnp.cos(aq)[None], jnp.sin(aq)[None])
        rk = rotate(k, jnp.cos(ak)[None], jnp.sin(ak)[None])
        return float(jnp.sum(rq * rk))

    assert np.isclose(score(3.0, 10.5), score(103.0, 110.5), atol=1e-4)
    assert not np.isclose(score(3.0, 10.5), score(3.0, 11.5), atol=1e-3)


def test_the_network_runs_and_its_velocity_depends_on_where_the_source_sits():
    gb = problem()
    sample = eqx.filter_jit(partial(train_sample, gb))(jr.key(0))
    net = LisaFlow(gb, 32, 2, 1, key=jr.key(1))
    params, static = jax.tree_util.tree_flatten(net)
    params = [
        (
            p + 0.3 * jr.normal(jr.key(i), p.shape, p.dtype)
            if isinstance(p, jax.Array) and jnp.issubdtype(p.dtype, jnp.floating)
            else p
        )
        for i, p in enumerate(params)
    ]
    net = jax.tree_util.tree_unflatten(static, params)
    delta, cache = eqx.filter_jit(net.encode)(sample.y, sample.f)
    frame = reference_frame(gb, delta, sample.f)
    z, t = sample.z0[0], sample.t[0]
    forward = eqx.filter_jit(net)
    out = forward(z, t, frame, sample.f, cache)
    assert out.shape == (*z.shape[:-1], 2 * z.shape[-1]) and bool(jnp.all(jnp.isfinite(out)))
    # another source's output sees the move through f0's RoPE angle and embedding
    moved = z.at[0, 0].add(0.5)
    assert not np.allclose(forward(moved, t, frame, sample.f, cache)[1], out[1], atol=1e-6)
    assert len(net.rope_periods) == 32 // 2 // 2 and net.rope_periods[0] == 2.0 / gb.t_obs
