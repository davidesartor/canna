"""canna.lisa.optim: the warmup-constant-cooldown schedule, the Muon/AdamW split and the optimizer."""

from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax

from canna.lisa import config
from canna.lisa.optim import is_muon, routing_report, schedule
from canna.lisa.train import setup, train_epoch, train_sample, train_step


def small_config(**overrides):
    network = dict(hidden_dim=32, num_blocks=2, num_heads=2, patch=16)
    return config.load("narrow-B", bf16="none", network=network, **overrides)


def test_the_schedule_warms_up_holds_and_cools_to_zero():
    lr = schedule(small_config(warmup_steps=10, total_steps=100, cooldown_frac=0.2), 2.0)
    at = lambda step: float(lr(jnp.asarray(step)))
    assert at(0) == 0.0 and np.isclose(at(5), 1.0)
    assert at(10) == at(50) == at(80) == 2.0
    assert np.isclose(at(90), 2.0 * (1 - np.sqrt(0.5)))
    assert at(100) == at(200) == 0.0
    held = schedule(small_config(warmup_steps=10, total_steps=100, cooldown_frac=0.0), 2.0)
    assert float(held(jnp.asarray(200))) == 2.0


def test_weight_decay_shrinks_the_muon_matrices_only():
    """F48: decay reaches Muon's matrices, never an AdamW leaf, matrix or not."""
    _, tx, state = setup(small_config(warmup_steps=1, weight_decay=50.0))
    params = eqx.filter(state.flow, eqx.is_inexact_array)
    zeros = jax.tree.map(jnp.zeros_like, params)
    p, opt_state, update = params, state.opt_state, eqx.filter_jit(tx.update)
    for _ in range(3):
        u, opt_state = update(zeros, opt_state, p)
        p = optax.apply_updates(p, u)

    leaves = jax.tree_util.tree_leaves_with_path(params)
    name = lambda path: ".".join(
        k.name for k in path if isinstance(k, jax.tree_util.GetAttrKey)
    )
    muon = {name(path): is_muon(path, leaf) for path, leaf in leaves}
    assert muon["backbone.blocks.mlp_y.linear2.weight"]
    assert muon["x_unembed.linear1.weight"]
    assert muon["frame_modulation.linear.weight"]
    # the 2-D AdamW leaves are what a bare ndim >= 2 rule would wrongly decay
    adamw = (
        "backbone.blocks.attention.qk_gain",
        "x_embed.weight",
        "frame_slot",
        "frame_unembed.weight",
        "x_unembed.linear2.weight",
        "x_unembed.linear2.bias",
    )
    assert not any(muon[n] for n in adamw)
    for (path, before), after in zip(leaves, jax.tree.leaves(p)):
        norm = lambda a: float(jnp.linalg.norm(a))
        if not is_muon(path, before):
            np.testing.assert_array_equal(after, before, err_msg=name(path))
        elif norm(before) > 0:
            assert norm(after) < 0.99 * norm(before), name(path)


def muon_decay_by_step(cfg, steps):
    """How much decay shrinks a Muon matrix at each of `steps` zero-gradient steps."""
    _, tx, state = setup(cfg)
    p = eqx.filter(state.flow, eqx.is_inexact_array)
    zeros = jax.tree.map(jnp.zeros_like, p)
    opt_state, norms, update = state.opt_state, [], eqx.filter_jit(tx.update)
    for _ in range(steps):
        u, opt_state = update(zeros, opt_state, p)
        p = optax.apply_updates(p, u)
        norms.append(float(jnp.linalg.norm(p.backbone.blocks.mlp_y.linear2.weight)))
    return np.diff(norms)


def test_no_cooldown_keeps_the_lr_past_total_steps_and_a_cooldown_stops_it():
    kwargs = dict(warmup_steps=1, weight_decay=1.0, total_steps=4)
    off = muon_decay_by_step(small_config(**kwargs), 8)
    on = muon_decay_by_step(small_config(cooldown_frac=0.5, **kwargs), 8)
    assert np.all(off < 0)
    assert np.all(on[:2] < 0) and np.all(on[4:] == 0)


def test_muon_and_adamw_moments_take_their_dtypes():
    for momentum_dtype in ("bfloat16", "float32"):
        _, _, state = setup(small_config(momentum_dtype=momentum_dtype))
        muon = [
            s for s in jax.tree.leaves(
                state.opt_state, is_leaf=lambda s: isinstance(s, optax.contrib.MuonState)
            )
            if isinstance(s, optax.contrib.MuonState)
        ]
        adam = [
            s for s in jax.tree.leaves(
                state.opt_state, is_leaf=lambda s: isinstance(s, optax.ScaleByAdamState)
            )
            if isinstance(s, optax.ScaleByAdamState)
        ]
        assert {a.dtype for s in muon for a in jax.tree.leaves(s.mu)} == {
            jnp.dtype(momentum_dtype)
        }
        assert {a.dtype for s in adam for a in jax.tree.leaves(s.mu)} == {
            jnp.dtype(jnp.float32)
        }


def test_float32_momentum_keeps_the_gradient_unrounded():
    cfg = small_config(momentum_dtype="float32")
    _, tx, state = setup(cfg)
    params = eqx.filter(state.flow, eqx.is_inexact_array)
    leaves, treedef = jax.tree.flatten(params)
    # small enough that the global clip leaves it alone
    grads = jax.tree.unflatten(
        treedef,
        [1e-4 * jr.normal(k, p.shape, p.dtype) / np.sqrt(p.size) for p, k in zip(leaves, jr.split(jr.key(0), len(leaves)))],
    )
    _, opt_state = tx.update(grads, state.opt_state, params)
    muon = next(
        s for s in jax.tree.leaves(
            opt_state, is_leaf=lambda s: isinstance(s, optax.contrib.MuonState)
        )
        if isinstance(s, optax.contrib.MuonState)
    )
    g = grads.backbone.blocks.mlp_y.linear2.weight
    np.testing.assert_allclose(
        muon.mu.backbone.blocks.mlp_y.linear2.weight,
        (1 - cfg.muon_momentum) * g,
        rtol=1e-6,
    )


def test_muon_orthogonalises_each_stacked_block_on_its_own():
    """Turning one block's gradient leaves another block's update alone: the leading axis is a batch axis."""
    network = dict(hidden_dim=32, num_blocks=3, num_heads=2, patch=16)
    _, tx, state = setup(
        config.load("narrow-B", bf16="none", network=network, warmup_steps=1)
    )
    params = eqx.filter(state.flow, eqx.is_inexact_array)
    leaves, treedef = jax.tree.flatten(params)
    grads = jax.tree.unflatten(
        treedef,
        [1e-4 * jr.normal(k, p.shape, p.dtype) / np.sqrt(p.size) for p, k in zip(leaves, jr.split(jr.key(0), len(leaves)))],
    )
    weight = grads.backbone.blocks.mlp_y.linear2.weight
    turned = eqx.tree_at(
        lambda g: g.backbone.blocks.mlp_y.linear2.weight,
        grads,
        weight.at[1].set(jr.permutation(jr.key(1), weight[1].ravel()).reshape(weight[1].shape)),
    )

    @eqx.filter_jit
    def update(g):
        # the first step's lr is 0 under warmup: take the second
        opt_state = tx.update(g, state.opt_state, params)[1]
        return tx.update(g, opt_state, params)[0].backbone.blocks.mlp_y.linear2.weight

    before, after = update(grads), update(turned)
    np.testing.assert_allclose(after[0], before[0], rtol=1e-5, atol=1e-9)
    assert not np.allclose(after[1], before[1], rtol=1e-2)


def test_routing_report_counts_every_parameter_once():
    _, _, state = setup(small_config())
    params = jax.tree.leaves(eqx.filter(state.flow, eqx.is_inexact_array))
    total = sum(p.size for p in params)
    assert f"of {total / 1e6:.2f}M parameters" in routing_report(state.flow)


def test_the_optimizer_builds_and_steps():

    problem, tx, state = setup(small_config())
    state, losses = train_epoch(
        problem, tx, state, 4, 3, draws=2
    )
    assert np.all(np.isfinite(np.asarray(losses)))


def test_micro_batches_take_the_full_batch_step():
    problem, tx, state = setup(small_config())
    keys = jr.split(jr.key(0), 4)
    batch = eqx.filter_jit(jax.vmap(partial(train_sample, problem)))(
        keys, jr.uniform(jr.key(1), (4, 2))
    )
    whole, whole_losses = train_step(problem, tx, state, batch)
    split, split_losses = train_step(problem, tx, state, batch, micro_batches=2)
    np.testing.assert_allclose(split_losses, whole_losses, rtol=1e-6)
    leaves = lambda s: jax.tree.leaves(eqx.filter(s.flow, eqx.is_inexact_array))
    for a, b in zip(leaves(split), leaves(whole)):
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-8)
