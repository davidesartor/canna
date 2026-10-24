"""Orbax checkpoints round-trip through restore_checkpoint and load_trained, optimizer state and histories included."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from canna.lisa import config
from canna.lisa.optim import eval_flow, routing_report
from canna.lisa.train import (
    checkpoint_manager,
    load_trained,
    restore_checkpoint,
    save_checkpoint,
    setup,
    train_epoch,
)

NETWORK = dict(hidden_dim=32, num_blocks=2, num_heads=2, patch=16)
HISTORIES = dict(loss_history=np.ones((5, 2, 2)), heldout_history=np.arange(5.0))


def small_config(**overrides):
    return config.load("narrow-B", bf16="none", network=NETWORK, **overrides)


def params(flow):
    return jax.tree.leaves(eqx.filter(flow, eqx.is_inexact_array))


def same(a, b):
    return all(jnp.array_equal(p, q) for p, q in zip(params(a), params(b)))


def moved_state(cfg):
    """A fresh state with the weights and the optimizer state moved off their initial values."""
    _, _, state = setup(cfg)
    return state._replace(
        flow=jax.tree.map(
            lambda p: p + 0.5 if eqx.is_inexact_array(p) else p, state.flow
        ),
        opt_state=jax.tree.map(
            lambda a: a + 1.0 if eqx.is_inexact_array(a) else a, state.opt_state
        ),
    )


def test_restore_brings_back_the_weights_the_optimizer_state_and_the_histories(
    tmp_path,
):
    cfg = small_config()
    state = moved_state(cfg)
    with checkpoint_manager(tmp_path) as checkpoints:
        assert checkpoints.latest_step() is None
        save_checkpoint(checkpoints, state, 3, HISTORIES)
    assert (tmp_path / "checkpoints" / "3").is_dir()

    with checkpoint_manager(tmp_path) as checkpoints:
        restored, epoch, histories = restore_checkpoint(
            checkpoints, setup(cfg)[2], tuple(HISTORIES)
        )
    assert epoch == 3 and histories["loss_history"].shape == (5, 2, 2)
    assert np.array_equal(histories["heldout_history"], np.arange(5.0))
    assert same(restored.flow, state.flow)
    opt = lambda s: [a for a in jax.tree.leaves(s.opt_state) if eqx.is_inexact_array(a)]
    assert all(jnp.array_equal(a, b) for a, b in zip(opt(restored), opt(state)))
    # and so the averaged weights the eval uses come back too
    assert same(
        eval_flow(restored.flow, restored.opt_state),
        eval_flow(state.flow, state.opt_state),
    )


def test_load_trained_picks_the_training_or_the_averaged_weights(tmp_path):
    cfg = small_config(output_dir=tmp_path, run="test-run")
    state = moved_state(cfg)
    with checkpoint_manager(cfg.run_dir) as checkpoints:
        save_checkpoint(checkpoints, state, 3, HISTORIES)

    _, training, epoch = load_trained(
        small_config(output_dir=tmp_path, run="test-run", weights="training")
    )
    _, average, _ = load_trained(
        small_config(output_dir=tmp_path, run="test-run", weights="average")
    )
    assert epoch == 3
    assert same(training, state.flow)
    assert same(average, eval_flow(state.flow, state.opt_state))
    assert not same(training, average)


def test_a_resumed_run_takes_the_same_steps_as_an_unbroken_one(tmp_path):
    problem, tx, state = setup(small_config())
    epoch = lambda s: train_epoch(problem, tx, s, 4, 1, draws=2)[0]
    unbroken = epoch(epoch(state))
    with checkpoint_manager(tmp_path) as checkpoints:
        save_checkpoint(checkpoints, epoch(state), 1, HISTORIES)
    with checkpoint_manager(tmp_path) as checkpoints:
        restored, _, _ = restore_checkpoint(
            checkpoints, setup(small_config())[2], tuple(HISTORIES)
        )
    resumed = epoch(restored)
    assert same(resumed.flow, unbroken.flow)
    assert jnp.array_equal(jax.random.key_data(resumed.key), jax.random.key_data(unbroken.key))


def test_only_the_latest_checkpoint_is_kept(tmp_path):
    state = moved_state(small_config())
    with checkpoint_manager(tmp_path) as checkpoints:
        for epoch in (1, 2):
            save_checkpoint(checkpoints, state, epoch, HISTORIES)
    assert [p.name for p in (tmp_path / "checkpoints").iterdir()] == ["2"]


def test_every_backbone_parameter_is_on_muon():
    _, _, state = setup(small_config())
    assert routing_report(state.flow).startswith("[optimizer] Muon")


def test_the_averaged_weights_start_at_the_initial_ones():
    _, _, state = setup(small_config())
    assert all(
        jnp.allclose(a, b)
        for a, b in zip(
            params(eval_flow(state.flow, state.opt_state)), params(state.flow)
        )
    )


def test_warm_start_copies_the_averaged_weights_but_a_redefined_head_and_lambda(tmp_path):
    from canna.lisa.train import warm_start

    load = lambda **kw: config.load("narrow-B", bf16="none", output_dir=tmp_path, **kw)
    source = load(run="source", network=NETWORK | dict(loss_weights=True))
    source.run_dir.mkdir(parents=True)
    config.save(source, source.run_dir / "config.yaml")
    state = moved_state(source)
    with checkpoint_manager(source.run_dir) as checkpoints:
        save_checkpoint(checkpoints, state, 2, {})
    averaged = eval_flow(state.flow, state.opt_state)

    target = load(network=NETWORK | dict(loss_weights=True, prediction="velocity"), init_from="source")
    fresh = setup(target)[2].flow
    started = warm_start(target, fresh)
    assert started.prediction == "velocity"
    assert same(started.backbone, averaged.backbone) and same(started.x_embed, averaged.x_embed)
    assert same(started.x_unembed.linear1, averaged.x_unembed.linear1)
    assert same(started.x_unembed.linear2, fresh.x_unembed.linear2)
    assert jnp.array_equal(started.loss_log_var, fresh.loss_log_var)
