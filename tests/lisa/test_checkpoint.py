"""Checkpoints with and without the optimizer state round-trip through restore_from."""

from pathlib import Path

import argparse
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import pytest
import yaml

import canna.lisa as lisa
from canna.lisa.train import TrainState


def small_args():
    with open(Path(lisa.__file__).parent / "configs" / "XS.yaml") as f:
        config = yaml.safe_load(f)
    config["network"] = dict(hidden_dim=32, num_blocks=1, num_heads=2, patch_stages=1)
    return argparse.Namespace(**config, dtype="float32", muon=True)


def params(state):
    return jax.tree.leaves(eqx.filter(state.flow, eqx.is_inexact_array))


@pytest.mark.parametrize("save_opt_state", [True, False])
def test_restore_brings_back_the_weights_with_or_without_optimizer_state(tmp_path, save_opt_state):
    args = small_args()
    state = TrainState.from_config(args)
    # move the weights and the optimizer state off their initial values
    state = state._replace(
        flow=jax.tree.map(lambda p: p + 0.5 if eqx.is_inexact_array(p) else p, state.flow),
        opt_state=jax.tree.map(lambda a: a + 1.0 if eqx.is_inexact_array(a) else a, state.opt_state),
    )
    manager = ocp.CheckpointManager(tmp_path, options=ocp.CheckpointManagerOptions(max_to_keep=1))
    state.save_to(manager, 3, np.zeros((5, 2, 3)), save_opt_state)
    manager.wait_until_finished()
    assert (tmp_path / "3" / "opt_state").exists() == save_opt_state

    fresh = TrainState.from_config(args)
    restored, epoch, hist = fresh.restore_from(manager)
    assert epoch == 3 and hist.shape == (5, 2, 3)
    assert all(jnp.array_equal(a, b) for a, b in zip(params(restored), params(state)))
    opt = lambda s: [a for a in jax.tree.leaves(s.opt_state) if eqx.is_inexact_array(a)]
    expected = opt(state) if save_opt_state else opt(fresh)
    assert all(jnp.array_equal(a, b) for a, b in zip(opt(restored), expected))
