"""from_config's args.Namespace shape, per train.py's __main__ argparse block --
the run yaml is loaded with set_defaults, so problem and network arrive as dicts."""

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest
import yaml
from flax import nnx

import canna
from canna.train import TrainState

CONFIG_ROOT = Path(canna.__file__).parent / "configs"


def _args(**overrides):
    with open(CONFIG_ROOT / "NoisyPoint-MLP-XS.yaml") as f:
        base = yaml.safe_load(f)
    base.update(dtype="float32", muon=False)
    base.update(overrides)
    return argparse.Namespace(**base)


def test_from_config_wires_problem_and_network_dims():
    """NoisyPoint's 2-D prior and MLP-XS's hidden_dim=128 should reach the built flow"""
    state = TrainState.from_config(_args())
    assert state.flow.x_embed.linear2.kernel.shape[-1] == 128


def test_from_config_metrics_start_at_zero():
    state = TrainState.from_config(_args())
    assert int(state.flow_metrics.count[...]) == 0
    assert int(state.x_metrics.count[...]) == 0
    assert int(state.y_metrics.count[...]) == 0


def test_from_config_same_seed_is_reproducible():
    """defect: identical args (same seed) must give identical network init, not
    time-seeded or otherwise nondeterministic params"""
    state_a = TrainState.from_config(_args(seed=7))
    state_b = TrainState.from_config(_args(seed=7))
    leaves_a = jax.tree.leaves(nnx.state(state_a.flow, nnx.Param))
    leaves_b = jax.tree.leaves(nnx.state(state_b.flow, nnx.Param))
    assert all(jnp.allclose(a, b) for a, b in zip(leaves_a, leaves_b))


def test_from_config_different_seed_changes_init():
    """the seed must actually reach the network init, not be ignored"""
    state_a = TrainState.from_config(_args(seed=1))
    state_b = TrainState.from_config(_args(seed=2))
    leaves_a = jax.tree.leaves(nnx.state(state_a.flow, nnx.Param))
    leaves_b = jax.tree.leaves(nnx.state(state_b.flow, nnx.Param))
    assert any(not jnp.allclose(a, b) for a, b in zip(leaves_a, leaves_b))


@pytest.mark.parametrize("muon", [True, False])
def test_from_config_muon_flag_builds_without_error(muon):
    """both optimizer branches (muon vs adamw) named in train.py's argparse must build"""
    state = TrainState.from_config(_args(muon=muon))
    assert state.optimizer is not None
