"""The __main__ epoch loop: sample_batch -> update_metrics -> aux_weight_schedule ->
weighted train_step -> periodic save_to. Exercised as a composition, since __main__ exposes
no single entry point for it."""

import jax.numpy as jnp
import orbax.checkpoint as ocp

from canna.problems import TrainSample
from canna.train import sample_batch, aux_weight_schedule


def test_one_epoch_composition_end_to_end(real_state, tmp_path):
    """mirrors train.py's __main__ body for a single epoch: the aux_weight-scaled,
    variance-rescaled weights must be finite and the resulting checkpoint must resume"""
    total_steps, warmup_frac = 10, 0.5
    batch = sample_batch(real_state.problem, real_state.rngs, 8)
    target_var = real_state.update_metrics(batch)
    aux_weight = aux_weight_schedule(0, total_steps, warmup_frac)
    weights = jnp.array([1.0, aux_weight, aux_weight]) / jnp.maximum(target_var, 1e-12)
    assert jnp.all(jnp.isfinite(weights))

    losses = real_state.train_step(batch, weights)
    assert losses.shape == (3,)

    manager = ocp.CheckpointManager(tmp_path)
    loss_hist = jnp.zeros((1, 1, 3)).at[0, 0].set(losses)
    real_state.save_to(manager, epoch=1, loss_hist=loss_hist)
    manager.wait_until_finished()

    epoch, restored_hist = real_state.restore_from(manager)
    assert epoch == 1
    assert jnp.allclose(restored_hist, loss_hist)


def test_zero_target_variance_stays_finite(real_state):
    """a batch whose targets are exactly constant gives zero variance; __main__'s
    jnp.maximum(target_var, 1e-12) guard is what keeps the division off inf"""
    n = 8
    degenerate = TrainSample(
        xt=jnp.zeros((n, 2)),
        dx=jnp.ones((n, 2)),
        t=jnp.full((n,), 0.5),
        y=jnp.zeros((n, 2)),
        x_target=jnp.full((n, 2), 3.0),
        y_target=jnp.ones((n, 2)),
    )
    target_var = real_state.update_metrics(degenerate)
    assert jnp.allclose(target_var, 0.0, atol=1e-6)
    weights = jnp.array([1.0, 1.0, 1.0]) / jnp.maximum(target_var, 1e-12)
    assert jnp.all(jnp.isfinite(weights))
