"""PositionedLisaFlow: source and token positions in bins, in one Fourier basis (F26)."""

import argparse
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
import yaml

import canna.lisa as lisa
from canna.lisa import LisaFlow, LisaGB, PositionedLisaFlow, train_sample
from canna.lisa.train import TrainState


def config(name):
    with open(Path(lisa.__file__).parent / "configs" / f"{name}.yaml") as f:
        return yaml.safe_load(f)


def problem(name):
    return LisaGB(**config(name)["problem"])


@pytest.mark.parametrize("name, atol", [("XS", 1e-4), ("B-late", 2e-3)])
def test_source_bins_match_the_problem_coordinate_map(name, atol):
    """The bin of f0 against a float64 copy of flow_to_physical's map, relative to the start."""
    gb = problem(name)
    windows = jax.vmap(gb.sample_f)(jr.split(jr.key(0), 64))
    x_f = jr.uniform(jr.key(1), (64, 16), minval=-1.0, maxval=1.0)
    got = jax.vmap(gb.window_geometry.source_bins)(x_f, windows)
    for f, x, b in zip(windows, np.asarray(x_f, np.float64), np.asarray(got)):
        start = int(gb.window_start(f))
        lo, hi = (float(v) for v in gb.f0_window(f))
        log_lo, log_hi = np.log(lo * gb.t_obs), np.log(hi * gb.t_obs)
        expected = np.exp((x * (log_hi - log_lo) + log_lo + log_hi) / 2) - start
        np.testing.assert_allclose(b, expected, atol=atol, rtol=0)


def test_source_bins_agree_with_flow_to_physical():
    """On XS, where float32 bin numbers are still sharp, against the problem itself."""
    gb = problem("XS")
    f = gb.sample_f(jr.key(2))
    x = gb.sample_flow(jr.key(3), f)
    f0_bins = gb.flow_to_physical(x, f)[:, 0] * gb.t_obs - gb.window_start(f)
    np.testing.assert_allclose(
        gb.window_geometry.source_bins(x[:, 0], f), f0_bins, atol=1e-2, rtol=0
    )


def test_every_b_token_gets_its_own_position_in_bf16():
    """The plain network's bf16 positions merge B's 2048 token columns into 641."""
    gb = problem("B-late")
    geometry = gb.window_geometry
    bins = geometry.token_bins(gb.window_bins // 2, 2)
    embed = eqx.nn.Linear(2 * 64, 512, key=jr.key(4))
    tokens = np.asarray(jax.vmap(embed)(geometry.features(bins, 64)).astype(jnp.bfloat16))
    assert len(np.unique(tokens, axis=0)) == gb.window_bins // 2
    old = np.asarray(jnp.arange(gb.window_bins // 2, dtype=jnp.bfloat16), np.float32)
    assert len(np.unique(old)) == 641


def test_features_span_two_bins_to_twice_the_window():
    geometry = problem("XS").window_geometry
    feats = geometry.features(jnp.asarray([0.0, 1.0]), 32)
    assert feats.shape == (2, 64)
    # the shortest period is 2 bins: one bin is half a turn
    np.testing.assert_allclose(feats[1, 0], 0.0, atol=1e-6)
    np.testing.assert_allclose(feats[1, 32], -1.0, atol=1e-6)


def positioned(dtype=jnp.float32):
    gb = problem("XS")
    sample = train_sample(gb, jr.key(5))
    net = PositionedLisaFlow(
        sample.xt.shape,
        sample.y.shape,
        32,
        2,
        1,
        positions=8,
        window=gb.window_geometry,
        dtype=dtype,
        key=jr.key(6),
    )
    # off init: zero-init gates make a fresh network ignore its conditioning
    params, static = eqx.partition(net, eqx.is_inexact_array)
    leaves, treedef = jax.tree.flatten(params)
    keys = jr.split(jr.key(7), len(leaves))
    leaves = [p + 0.3 * jr.normal(k, p.shape, p.dtype) for p, k in zip(leaves, keys)]
    return gb, sample, eqx.combine(jax.tree.unflatten(treedef, leaves), static)


def test_positioned_flow_is_permutation_equivariant_over_sources():
    _, sample, net = positioned()
    perm = jnp.array([2, 0, 3, 1])
    dx, x, _ = net(sample.xt, sample.t, sample.y, sample.f)
    dx_p, x_p, _ = net(sample.xt[perm], sample.t, sample.y, sample.f)
    np.testing.assert_allclose(dx_p, dx[perm], atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(x_p, x[perm], atol=1e-4, rtol=1e-4)


def test_positioned_flow_sees_a_sub_bin_shift_in_bf16():
    """Move one source by a tenth of a bin: in bf16 the plain coordinate is ~0.1 bins coarse."""
    gb, sample, net = positioned(jnp.bfloat16)
    bins_per_unit = (gb.window_bins - 2 * gb.window_geometry.guard) / 2
    shifted = sample.xt.at[0, 0].add(0.1 / bins_per_unit)
    dx = net(sample.xt, sample.t, sample.y, sample.f)[0]
    dx_s = net(shifted, sample.t, sample.y, sample.f)[0]
    assert not np.array_equal(np.asarray(dx[0]), np.asarray(dx_s[0]))


def run_args(name, **network):
    cfg = config(name)
    cfg["network"] = dict(cfg["network"], hidden_dim=32, num_blocks=1, num_heads=2, **network)
    return argparse.Namespace(**cfg, dtype="float32", muon=True)


def test_config_picks_the_network_class():
    assert type(TrainState.from_config(run_args("XS-late")).flow) is LisaFlow
    state = TrainState.from_config(run_args("XS-pos"))
    assert type(state.flow) is PositionedLisaFlow and state.flow.positions == 32
    sample = train_sample(state.problem, jr.key(8))
    dx, _, _ = state.flow(sample.xt, sample.t, sample.y, sample.f)
    assert dx.shape == sample.xt.shape and bool(jnp.all(jnp.isfinite(dx)))


def test_positioned_checkpoint_round_trips(tmp_path):
    import orbax.checkpoint as ocp

    args = run_args("XS-pos")
    state = TrainState.from_config(args)
    state = state._replace(
        flow=jax.tree.map(lambda p: p + 0.5 if eqx.is_inexact_array(p) else p, state.flow)
    )
    manager = ocp.CheckpointManager(tmp_path, options=ocp.CheckpointManagerOptions(max_to_keep=1))
    state.save_to(manager, 2, np.zeros((3, 2, 3)), False)
    manager.wait_until_finished()
    restored, epoch, _ = TrainState.from_config(args).restore_from(manager)
    leaves = lambda s: jax.tree.leaves(eqx.filter(s.flow, eqx.is_inexact_array))
    assert epoch == 2
    assert all(jnp.array_equal(a, b) for a, b in zip(leaves(restored), leaves(state)))
