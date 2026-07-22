"""save_to/restore_from must let an interrupted run resume exactly where it stopped."""

import jax
import jax.numpy as jnp
from flax import nnx
import orbax.checkpoint as ocp

from canna.train import sample_batch


def _param_leaves(module):
    return jax.tree.leaves(jax.tree.map(jnp.array, nnx.state(module, nnx.Param)))


def test_roundtrip_restores_exact_flow_params(tmp_path, make_state):
    state = make_state(seed=11)
    manager = ocp.CheckpointManager(tmp_path)
    loss_hist = jnp.arange(2 * 3 * 3, dtype=jnp.float32).reshape(2, 3, 3)
    state.save_to(manager, epoch=1, loss_hist=loss_hist)
    manager.wait_until_finished()

    fresh = make_state(seed=999)
    manager2 = ocp.CheckpointManager(tmp_path)
    epoch, restored_hist = fresh.restore_from(manager2)

    assert epoch == 1
    assert jnp.allclose(restored_hist, loss_hist)
    orig = _param_leaves(state.flow)
    restored = _param_leaves(fresh.flow)
    assert all(jnp.allclose(o, r) for o, r in zip(orig, restored))


def test_roundtrip_restores_metric_running_stats(tmp_path, make_state, fake_problem):
    """resume must also preserve the Welford running stats, not just model weights --
    they drive the loss-rescaling weights the next step would use"""
    state = make_state(seed=21)
    batch = sample_batch(fake_problem, nnx.Rngs(22), 4)
    state.update_metrics(batch)

    manager = ocp.CheckpointManager(tmp_path)
    state.save_to(manager, epoch=3, loss_hist=jnp.zeros((1, 1, 3)))
    manager.wait_until_finished()

    fresh = make_state(seed=23)
    manager2 = ocp.CheckpointManager(tmp_path)
    fresh.restore_from(manager2)

    assert int(fresh.flow_metrics.count[...]) == int(state.flow_metrics.count[...])
    assert jnp.allclose(fresh.flow_metrics.mean[...], state.flow_metrics.mean[...])


def test_roundtrip_restores_optimizer_momentum(tmp_path, make_state, fake_problem):
    """defect: adam's first/second moment estimates must survive a restore, or resumed
    training silently restarts optimizer momentum from zero"""
    state = make_state(seed=31)
    batch = sample_batch(fake_problem, nnx.Rngs(32), 4)
    state.train_step(batch, jnp.ones((3,)))

    manager = ocp.CheckpointManager(tmp_path)
    state.save_to(manager, epoch=1, loss_hist=jnp.zeros((1, 1, 3)))
    manager.wait_until_finished()

    fresh = make_state(seed=999)
    manager2 = ocp.CheckpointManager(tmp_path)
    fresh.restore_from(manager2)

    orig_opt = jax.tree.leaves(jax.tree.map(jnp.array, nnx.state(state.optimizer)))
    restored_opt = jax.tree.leaves(jax.tree.map(jnp.array, nnx.state(fresh.optimizer)))
    assert len(orig_opt) == len(restored_opt)
    assert all(jnp.allclose(o, r) for o, r in zip(orig_opt, restored_opt))


def test_restore_from_empty_manager_returns_no_history(tmp_path, make_state):
    """a fresh run with no checkpoints yet resumes from epoch 0 with no loss history"""
    state = make_state(seed=12)
    manager = ocp.CheckpointManager(tmp_path)
    epoch, loss_hist = state.restore_from(manager)
    assert loss_hist is None
    assert epoch == 0
