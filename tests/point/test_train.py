"""The point trainer: running metrics, one step, one epoch, resume."""

import argparse
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import orbax.checkpoint as ocp
import pytest
import yaml

import canna.point as point
from canna.point import TrainSample
from canna.point.train import TrainState, parse_args

CONFIG_ROOT = Path(point.__file__).parent / "configs"
DIM = 2


def args(**overrides):
    with open(CONFIG_ROOT / "XS.yaml") as f:
        base = yaml.safe_load(f)
    base.update(
        dtype="float32",
        muon=False,
        network=dict(hidden_dim=8, num_blocks=1),
        config="XS",
        config_root=CONFIG_ROOT,
        output_dir=Path("outputs"),
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def batch(problem, key, n=4):
    return jax.vmap(problem.train_sample)(jr.split(key, n))


def param_leaves(module):
    return jax.tree.leaves(eqx.filter(module, eqx.is_inexact_array))


def opt_step(state):
    return int(optax.tree_utils.tree_get(state.opt_state, "count"))


def constant_batch(n=8, dx=1.0):
    return TrainSample(
        xt=jnp.zeros((n, DIM)),
        dx=jnp.full((n, DIM), dx),
        t=jnp.full((n,), 0.5),
        y=jnp.zeros((n, DIM)),
    )


# --- from_config -----------------------------------------------------------


def test_from_config_shapes_the_network_from_one_problem_sample():
    state = TrainState.from_config(args())
    assert state.flow.x_embed.linear2.weight.shape[0] == 8
    assert state.flow.x_unembed.linear2.weight.shape[0] == DIM


def test_from_config_metrics_start_empty():
    state = TrainState.from_config(args())
    assert int(state.flow_metrics.count) == 0


def test_from_config_is_reproducible_under_one_seed():
    a = param_leaves(TrainState.from_config(args(seed=7)).flow)
    b = param_leaves(TrainState.from_config(args(seed=7)).flow)
    assert all(jnp.allclose(u, v) for u, v in zip(a, b))


def test_from_config_seed_reaches_the_network_init():
    a = param_leaves(TrainState.from_config(args(seed=1)).flow)
    b = param_leaves(TrainState.from_config(args(seed=2)).flow)
    assert any(not jnp.allclose(u, v) for u, v in zip(a, b))


@pytest.mark.parametrize("muon", [True, False])
def test_from_config_builds_both_optimizer_branches(muon):
    state = TrainState.from_config(args(muon=muon))
    assert state.tx is not None and state.opt_state is not None


def test_parse_args_defaults_come_from_the_run_config(monkeypatch):
    monkeypatch.setattr("sys.argv", ["train.py", "--config", "XS"])
    parsed = parse_args()
    with open(CONFIG_ROOT / "XS.yaml") as f:
        config = yaml.safe_load(f)
    assert parsed.batch_size == config["batch_size"]
    assert parsed.problem == config["problem"]


def test_parse_args_cli_flag_overrides_the_run_config(monkeypatch):
    monkeypatch.setattr(
        "sys.argv", ["train.py", "--config", "XS", "--batch_size", "17"]
    )
    assert parse_args().batch_size == 17


# --- running metrics -------------------------------------------------------


def test_train_step_updates_the_running_variance():
    state = TrainState.from_config(args())
    state, _ = state.train_step(batch(state.problem, jr.key(1), 16))
    assert int(state.flow_metrics.count) > 0


def test_constant_targets_give_exactly_zero_variance():
    state = TrainState.from_config(args())
    state, _ = state.train_step(constant_batch())
    assert jnp.allclose(state.flow_metrics.variance, 0.0, atol=1e-6)


def test_the_metric_tracks_the_flow_target():
    n = 32
    b = TrainSample(
        xt=jnp.zeros((n, DIM)),
        dx=jnp.arange(n * DIM, dtype=jnp.float32).reshape(n, DIM),
        t=jnp.full((n,), 0.5),
        y=jnp.zeros((n, DIM)),
    )
    state = TrainState.from_config(args())
    state, _ = state.train_step(b)
    assert float(state.flow_metrics.variance) == pytest.approx(
        float(jnp.var(b.dx)), rel=1e-3
    )


def test_metrics_accumulate_across_jitted_calls():
    """the Welford carried in the returned state must keep the earlier count"""
    state = TrainState.from_config(args())
    state, _ = state.train_step(constant_batch(n=4, dx=-1.0))
    first = int(state.flow_metrics.count)
    second_batch = constant_batch(n=4, dx=1.0)
    state, _ = state.train_step(second_batch)
    assert first > 0
    assert int(state.flow_metrics.count) == first + second_batch.dx.size
    assert state.flow_metrics.variance > 0.0


def test_zero_target_variance_keeps_the_loss_finite():
    """the jnp.maximum(variance, 1e-12) guard is what keeps the weight off inf"""
    state = TrainState.from_config(args())
    state, loss = state.train_step(constant_batch())
    assert jnp.allclose(state.flow_metrics.variance, 0.0, atol=1e-6)
    assert jnp.all(jnp.isfinite(loss))


# --- train_step ------------------------------------------------------------


def test_train_step_moves_some_parameter():
    state = TrainState.from_config(args())
    before = param_leaves(state.flow)
    state, _ = state.train_step(batch(state.problem, jr.key(3)))
    after = param_leaves(state.flow)
    assert any(not jnp.allclose(b, a) for b, a in zip(before, after))


def test_train_step_returns_a_finite_scalar_loss():
    state = TrainState.from_config(args())
    _, loss = state.train_step(batch(state.problem, jr.key(4)))
    assert loss.shape == ()
    assert jnp.all(jnp.isfinite(loss))


def test_reported_loss_is_raw_not_variance_weighted():
    """the returned aux is the unweighted mse, so a large-target batch reports
    a large loss even though the weight scales it down inside the gradient"""
    state = TrainState.from_config(args())
    _, loss = state.train_step(constant_batch(dx=1000.0))
    assert float(loss) > 1.0


def test_train_step_advances_the_optimizer():
    state = TrainState.from_config(args())
    before_step = opt_step(state)
    before_params = param_leaves(state.flow)
    state, _ = state.train_step(batch(state.problem, jr.key(10)))
    assert opt_step(state) == before_step + 1
    assert any(
        not jnp.allclose(b, a) for b, a in zip(before_params, param_leaves(state.flow))
    )


# --- train_epoch and checkpointing -----------------------------------------


def test_train_epoch_returns_one_loss_per_step():
    state = TrainState.from_config(args())
    state, losses = state.train_epoch(batch_size=8, n_steps=3)
    assert losses.shape == (3,)
    assert jnp.all(jnp.isfinite(losses))
    assert opt_step(state) == 3


def test_one_epoch_then_checkpoint_resumes(tmp_path):
    state = TrainState.from_config(args())
    state, losses = state.train_epoch(batch_size=8, n_steps=3)

    manager = ocp.CheckpointManager(tmp_path)
    loss_hist = jnp.zeros((1, 3)).at[0].set(losses)
    state.save_to(manager, epoch=1, loss_hist=loss_hist)
    manager.wait_until_finished()

    _, epoch, restored = state.restore_from(manager)
    assert epoch == 1
    assert jnp.allclose(restored, loss_hist)


def test_roundtrip_restores_exact_flow_params(tmp_path):
    state = TrainState.from_config(args(seed=11))
    manager = ocp.CheckpointManager(tmp_path)
    loss_hist = jnp.arange(2 * 3, dtype=jnp.float32).reshape(2, 3)
    state.save_to(manager, epoch=1, loss_hist=loss_hist)
    manager.wait_until_finished()

    fresh = TrainState.from_config(args(seed=999))
    fresh, epoch, restored_hist = fresh.restore_from(ocp.CheckpointManager(tmp_path))
    assert epoch == 1
    assert jnp.allclose(restored_hist, loss_hist)
    assert all(
        jnp.allclose(o, r)
        for o, r in zip(param_leaves(state.flow), param_leaves(fresh.flow))
    )


def test_roundtrip_restores_the_running_metric_stats(tmp_path):
    """they drive the loss rescaling the next step would use, so a resume must keep them"""
    state = TrainState.from_config(args(seed=21))
    state, _ = state.train_step(batch(state.problem, jr.key(22)))
    manager = ocp.CheckpointManager(tmp_path)
    state.save_to(manager, epoch=3, loss_hist=jnp.zeros((1, 1)))
    manager.wait_until_finished()

    fresh = TrainState.from_config(args(seed=23))
    fresh, *_ = fresh.restore_from(ocp.CheckpointManager(tmp_path))
    assert int(fresh.flow_metrics.count) == int(state.flow_metrics.count)
    assert jnp.allclose(fresh.flow_metrics.mean, state.flow_metrics.mean)


def test_roundtrip_restores_the_prng_key(tmp_path):
    """without this a resumed run redraws the batches it already trained on"""
    state = TrainState.from_config(args(seed=41))
    state, _ = state.train_epoch(batch_size=8, n_steps=2)
    manager = ocp.CheckpointManager(tmp_path)
    state.save_to(manager, epoch=1, loss_hist=jnp.zeros((1, 2)))
    manager.wait_until_finished()

    fresh = TrainState.from_config(args(seed=999))
    fresh, *_ = fresh.restore_from(ocp.CheckpointManager(tmp_path))
    assert jnp.array_equal(jr.key_data(fresh.key), jr.key_data(state.key))


def test_roundtrip_restores_optimizer_momentum(tmp_path):
    """without this a resumed run silently restarts momentum from zero"""
    state = TrainState.from_config(args(seed=31))
    state, _ = state.train_step(batch(state.problem, jr.key(32)))
    manager = ocp.CheckpointManager(tmp_path)
    state.save_to(manager, epoch=1, loss_hist=jnp.zeros((1, 1)))
    manager.wait_until_finished()

    fresh = TrainState.from_config(args(seed=999))
    fresh, *_ = fresh.restore_from(ocp.CheckpointManager(tmp_path))
    orig = jax.tree.leaves(eqx.filter(state.opt_state, eqx.is_array))
    restored = jax.tree.leaves(eqx.filter(fresh.opt_state, eqx.is_array))
    assert len(orig) == len(restored)
    assert all(jnp.allclose(o, r) for o, r in zip(orig, restored))


def test_restore_from_an_empty_manager_starts_at_epoch_zero(tmp_path):
    state = TrainState.from_config(args())
    _, epoch, loss_hist = state.restore_from(ocp.CheckpointManager(tmp_path))
    assert epoch == 0 and loss_hist is None
