"""train_step takes one optimizer step on the weighted sum of the three losses."""

import jax
import jax.numpy as jnp
from flax import nnx

from canna.train import sample_batch


def _param_leaves(module):
    return jax.tree.leaves(jax.tree.map(jnp.array, nnx.state(module, nnx.Param)))


def test_zero_weights_leave_flow_unchanged(tiny_state, fake_problem):
    """weights=[0,0,0] must be a true no-op step -- zero gradient, zero update"""
    batch = sample_batch(fake_problem, nnx.Rngs(2), 4)
    before = _param_leaves(tiny_state.flow)
    tiny_state.train_step(batch, jnp.zeros((3,)))
    after = _param_leaves(tiny_state.flow)
    assert all(jnp.allclose(b, a) for b, a in zip(before, after))


def test_full_weights_move_some_parameter(tiny_state, fake_problem):
    """some leaf must move under weights=[1,1,1]; per CLAUDE.md, zero-init Modulation
    keeps gated branches at identity at init, so we assert *some* movement, not all"""
    batch = sample_batch(fake_problem, nnx.Rngs(3), 4)
    before = _param_leaves(tiny_state.flow)
    tiny_state.train_step(batch, jnp.ones((3,)))
    after = _param_leaves(tiny_state.flow)
    assert any(not jnp.allclose(b, a) for b, a in zip(before, after))


def test_returned_losses_are_finite_triplet(tiny_state, fake_problem):
    batch = sample_batch(fake_problem, nnx.Rngs(4), 4)
    losses = tiny_state.train_step(batch, jnp.ones((3,)))
    assert losses.shape == (3,)
    assert jnp.all(jnp.isfinite(losses))


def test_returned_losses_independent_of_weights(make_state, fake_problem):
    """train_step reports raw per-term losses, not pre-weighted ones"""
    batch = sample_batch(fake_problem, nnx.Rngs(5), 4)
    state_a = make_state(seed=42)
    state_b = make_state(seed=42)
    losses_a = state_a.train_step(batch, jnp.array([1.0, 0.0, 0.0]))
    losses_b = state_b.train_step(batch, jnp.array([0.0, 0.0, 1.0]))
    assert jnp.allclose(losses_a, losses_b, atol=1e-5)


def test_y_unembed_untouched_when_y_weight_zero(make_state, fake_problem):
    """defect: y_unembed only feeds the y_target loss (MLPFlow.__call__ returns y_unembed(y)
    used solely as y_target); with weights=[1,1,0] its params must not move"""
    state = make_state(seed=7)
    batch = sample_batch(fake_problem, nnx.Rngs(6), 4)
    before = _param_leaves(state.flow.y_unembed)
    state.train_step(batch, jnp.array([1.0, 1.0, 0.0]))
    after = _param_leaves(state.flow.y_unembed)
    assert all(jnp.allclose(b, a) for b, a in zip(before, after))


def test_y_unembed_moves_when_only_y_weight_set(make_state, fake_problem):
    """defect: confirms weights index 2 actually gates the y term (not a no-op / index swap)"""
    state = make_state(seed=8)
    batch = sample_batch(fake_problem, nnx.Rngs(9), 4)
    before = _param_leaves(state.flow.y_unembed)
    state.train_step(batch, jnp.array([0.0, 0.0, 1.0]))
    after = _param_leaves(state.flow.y_unembed)
    assert any(not jnp.allclose(b, a) for b, a in zip(before, after))


def test_mutation_persists_and_optimizer_advances(real_state):
    """params AND optimizer momentum must both mutate in place through the jit+NamedTuple
    boundary; a broken graph-update would leave one or the other stale"""
    batch = sample_batch(real_state.problem, real_state.rngs, 4)
    opt_step_before = int(real_state.optimizer.step[...])
    params_before = _param_leaves(real_state.flow)

    real_state.train_step(batch, jnp.ones((3,)))

    opt_step_after = int(real_state.optimizer.step[...])
    params_after = _param_leaves(real_state.flow)
    assert opt_step_after == opt_step_before + 1
    assert any(not jnp.allclose(b, a) for b, a in zip(params_before, params_after))
