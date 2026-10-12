"""Checkpoints round-trip through restore_from, optimizer state and histories included."""

from pathlib import Path

import argparse
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import yaml

import canna.lisa as lisa
from canna.lisa.train import TrainState, eval_flow


def small_args():
    with open(Path(lisa.__file__).parent / "configs" / "XS-fd.yaml") as f:
        config = yaml.safe_load(f)
    config["network"] = dict(hidden_dim=32, num_blocks=1, num_heads=2, patch=2)
    return argparse.Namespace(**config, dtype="float32")


def params(flow):
    return jax.tree.leaves(eqx.filter(flow, eqx.is_inexact_array))


def test_restore_brings_back_the_weights_the_optimizer_state_and_the_histories(tmp_path):
    args = small_args()
    state = TrainState.from_config(args)
    # move the weights and the optimizer state off their initial values
    state = state._replace(
        flow=jax.tree.map(lambda p: p + 0.5 if eqx.is_inexact_array(p) else p, state.flow),
        opt_state=jax.tree.map(lambda a: a + 1.0 if eqx.is_inexact_array(a) else a, state.opt_state),
    )
    manager = ocp.CheckpointManager(tmp_path, options=ocp.CheckpointManagerOptions(max_to_keep=1))
    state.save_to(manager, 3, np.ones((5, 2, 2)), np.arange(5.0))
    manager.wait_until_finished()

    restored, epoch, (hist, heldout) = TrainState.from_config(args).restore_from(manager)
    assert epoch == 3 and hist.shape == (5, 2, 2)
    assert np.array_equal(heldout, np.arange(5.0))
    assert all(jnp.array_equal(a, b) for a, b in zip(params(restored.flow), params(state.flow)))
    opt = lambda s: [a for a in jax.tree.leaves(s.opt_state) if eqx.is_inexact_array(a)]
    assert all(jnp.array_equal(a, b) for a, b in zip(opt(restored), opt(state)))
    # and so the averaged weights the eval uses come back too
    assert all(
        jnp.array_equal(a, b)
        for a, b in zip(
            params(eval_flow(restored.flow, restored.opt_state)),
            params(eval_flow(state.flow, state.opt_state)),
        )
    )


def test_every_backbone_parameter_is_on_muon():
    report = TrainState.from_config(small_args()).routing_report()
    assert report.startswith("[optimizer] Muon")


def test_the_averaged_weights_start_at_the_initial_ones():
    state = TrainState.from_config(small_args())
    assert all(
        jnp.allclose(a, b)
        for a, b in zip(params(eval_flow(state.flow, state.opt_state)), params(state.flow))
    )
