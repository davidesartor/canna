"""RoPE in frequency: the sources' and tokens' bins, the rotation, and the network with it."""

from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import yaml

import canna.lisa as lisa
from canna.lisa import LisaFlow, LisaGB
from canna.networks.mmdit import rotate
from ._helpers import window


def problem(name="XS-fd"):
    with open(Path(lisa.__file__).parent / "configs" / f"{name}.yaml") as f:
        return LisaGB(**yaml.safe_load(f)["problem"])


def test_source_bins_agree_with_flow_to_physical():
    for name, atol in (("XS-fd", 1e-3), ("B-fd", 2e-2)):
        gb = problem(name)
        f = window(gb)
        x = gb.sample_flow(jr.key(1), f)
        f0_bins = gb.flow_to_physical(x, f)[:, 0] * gb.t_obs - gb.window_start(f)
        np.testing.assert_allclose(gb.window_geometry.source_bins(x[:, 0], f), f0_bins, atol=atol, rtol=0)


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


def test_a_rope_network_runs_and_its_velocity_depends_on_where_the_source_sits():
    gb = problem()
    sample = lisa.train_sample(gb, jr.key(0))
    net = LisaFlow(
        sample.xt.shape, sample.y.shape, 32, 2, 1, gb.f0_range, 2,
        rope=True, window=gb.window_geometry, key=jr.key(1),
    )
    params, static = jax.tree_util.tree_flatten(net)
    params = [p + 0.3 * jr.normal(jr.key(i), p.shape, p.dtype) if isinstance(p, jax.Array) and jnp.issubdtype(p.dtype, jnp.floating) else p for i, p in enumerate(params)]
    net = jax.tree_util.tree_unflatten(static, params)
    dx, x1 = net(sample.xt, sample.t, sample.y, sample.f)
    assert dx.shape == sample.xt.shape and bool(jnp.all(jnp.isfinite(dx)))
    moved = sample.xt.at[0, 0].add(0.05)
    assert not np.allclose(net(moved, sample.t, sample.y, sample.f)[0][1], dx[1], atol=1e-6)
    assert len(net.rope_periods) == 32 // 2 // 2 and net.rope_periods[0] == 2.0
