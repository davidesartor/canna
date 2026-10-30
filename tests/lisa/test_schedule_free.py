"""canna.lisa.optim.schedule_free: optax's schedule-free when the fixes are off, each fix (F37), and the optimizer."""

from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax

from canna.lisa import config
from canna.lisa.optim import (
    interpolation,
    is_muon,
    normuon_rows,
    routing_report,
    schedule_free,
    schedule_free_state,
)
from canna.lisa.train import setup, train_epoch, train_sample, train_step

TARGET = jnp.array([[1.0, -2.0], [0.5, 3.0]])


def small_config(**overrides):
    network = dict(hidden_dim=32, num_blocks=2, num_heads=2, patch=16)
    return config.load("narrow-B", bf16="none", network=network, **overrides)


def grad(p):
    return jax.grad(lambda q: 0.5 * jnp.sum(jnp.square(q["w"] - TARGET)))(p)


def run(tx, steps, params=None):
    params = params or {"w": jnp.zeros((2, 2))}
    state = tx.init(params)
    for _ in range(steps):
        u, state = tx.update(grad(params), state, params)
        params = optax.apply_updates(params, u)
    return params, state


def test_matches_optax_schedule_free_with_the_fixes_off():
    lr = 0.1
    ours = schedule_free(
        optax.sgd(lr), lambda s: lr, beta_start=0.9, beta_end=0.9, warm_start=0, r=0.0
    )
    reference = optax.contrib.schedule_free(optax.sgd(lr), lr, b1=0.9)
    y1, s1 = run(ours, 60)
    y2, s2 = run(reference, 60)
    np.testing.assert_allclose(y1["w"], y2["w"], rtol=1e-5, atol=1e-6)
    x2 = optax.contrib.schedule_free_eval_params(s2, y2)
    np.testing.assert_allclose(s1.x["w"], x2["w"], rtol=1e-5, atol=1e-6)


def test_decay_on_z_holds_a_constantly_pushed_z_at_its_fixed_point():
    push = optax.GradientTransformation(
        lambda p: optax.EmptyState(),
        lambda g, s, p=None: (jax.tree.map(lambda a: 0.01 + 0 * a, g), s),
    )
    lr, lam = 0.1, 0.5
    free = schedule_free(push, lambda s: lr, weight_decay=0.0)
    held = schedule_free(push, lambda s: lr, weight_decay=lam)
    _, s_free = run(free, 2000)
    _, s_held = run(held, 2000)
    # without decay z walks off linearly; with it, z -> u / (lr lambda)
    assert np.allclose(s_free.z["w"], 20.0, rtol=1e-3)
    np.testing.assert_allclose(s_held.z["w"], 0.01 / (lr * lam), rtol=1e-3)


def test_x_follows_z_until_the_warm_start():
    tx = schedule_free(optax.sgd(0.1), lambda s: 0.1, warm_start=10)
    for steps in (1, 5, 10):
        _, s = run(tx, steps)
        np.testing.assert_allclose(s.x["w"], s.z["w"])
    _, s = run(tx, 20)
    assert not np.allclose(s.x["w"], s.z["w"])


def test_late_iterates_weigh_more_with_r_one():
    # on a target reached early, r=1 pulls x towards the late iterates faster than r=0
    uniform = schedule_free(optax.sgd(0.05), lambda s: 0.05, r=0.0)
    late = schedule_free(optax.sgd(0.05), lambda s: 0.05, r=1.0)
    _, s0 = run(uniform, 200)
    _, s1 = run(late, 200)
    err = lambda s: float(jnp.linalg.norm(s.x["w"] - TARGET))
    assert err(s1) < err(s0)


def test_interpolation_runs_from_start_to_end():
    b = lambda t: float(interpolation(jnp.asarray(t), 0.9, 0.965, 100, 10_000))
    assert b(1) == b(100) == np.float32(0.9)
    assert np.isclose(b(10_000), 0.965) and np.isclose(b(50_000), 0.965)
    values = [b(t) for t in (100, 300, 1000, 3000, 10_000)]
    assert all(a < c for a, c in zip(values, values[1:]))


def test_normuon_rows_keeps_each_matrix_norm_and_evens_out_its_rows():

    _, _, state = setup(small_config())
    flow = state.flow
    params = eqx.filter(flow, eqx.is_inexact_array)
    # rows of very different sizes in every Muon matrix
    updates = jax.tree.map(
        lambda p: (
            p * 0 + jnp.arange(1.0, 1.0 + p.shape[-2])[:, None]
            if p.ndim >= 2
            else p * 0 + 1.0
        ),
        params,
    )
    tx = normuon_rows()
    out, _ = tx.update(updates, tx.init(params), params)
    u = updates.backbone.blocks.mlp_y.linear2.weight
    o = out.backbone.blocks.mlp_y.linear2.weight
    np.testing.assert_allclose(jnp.linalg.norm(o[0]), jnp.linalg.norm(u[0]), rtol=1e-4)
    rows_u, rows_o = jnp.linalg.norm(u[0], axis=-1), jnp.linalg.norm(o[0], axis=-1)
    assert (
        float(rows_o.max() / rows_o.min()) < 1.01 < float(rows_u.max() / rows_u.min())
    )
    # AdamW's leaves are left alone
    np.testing.assert_allclose(
        out.x_unembed.linear2.weight, updates.x_unembed.linear2.weight
    )


def test_weight_decay_shrinks_the_muon_matrices_only():
    """F48: decay on z reaches Muon's matrices, never an AdamW leaf, matrix or not."""
    _, tx, state = setup(small_config(warmup_steps=1, weight_decay=50.0))
    params = eqx.filter(state.flow, eqx.is_inexact_array)
    zeros = jax.tree.map(jnp.zeros_like, params)
    p, opt_state, update = params, state.opt_state, eqx.filter_jit(tx.update)
    for _ in range(3):
        u, opt_state = update(zeros, opt_state, p)
        p = optax.apply_updates(p, u)
    z = schedule_free_state(opt_state).z

    leaves = jax.tree_util.tree_leaves_with_path(params)
    name = lambda path: ".".join(
        k.name for k in path if isinstance(k, jax.tree_util.GetAttrKey)
    )
    muon = {name(path): is_muon(path, leaf) for path, leaf in leaves}
    assert muon["backbone.blocks.mlp_y.linear2.weight"]
    assert muon["x_unembed.linear1.weight"]
    # the 2-D AdamW leaves are what a bare ndim >= 2 rule would wrongly decay
    adamw = (
        "backbone.blocks.attention.qk_gain",
        "x_embed.linear.weight",
        "x_unembed.linear2.weight",
        "x_unembed.linear2.bias",
    )
    assert not any(muon[n] for n in adamw)
    for (path, before), after in zip(leaves, jax.tree.leaves(z)):
        norm = lambda a: float(jnp.linalg.norm(a))
        if not is_muon(path, before):
            np.testing.assert_array_equal(after, before, err_msg=name(path))
        elif norm(before) > 0:
            assert norm(after) < 0.99 * norm(before), name(path)


def muon_decay_by_step(cfg, steps):
    """How much decay on z shrinks a Muon matrix at each of `steps` zero-gradient steps."""
    _, tx, state = setup(cfg)
    p = eqx.filter(state.flow, eqx.is_inexact_array)
    zeros = jax.tree.map(jnp.zeros_like, p)
    opt_state, norms, update = state.opt_state, [], eqx.filter_jit(tx.update)
    for _ in range(steps):
        u, opt_state = update(zeros, opt_state, p)
        p = optax.apply_updates(p, u)
        z = schedule_free_state(opt_state).z
        norms.append(float(jnp.linalg.norm(z.backbone.blocks.mlp_y.linear2.weight)))
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
    whole, whole_losses = train_step(tx, state, batch)
    split, split_losses = train_step(tx, state, batch, micro_batches=2)
    np.testing.assert_allclose(split_losses, whole_losses, rtol=1e-6)
    leaves = lambda s: jax.tree.leaves(eqx.filter(s.flow, eqx.is_inexact_array))
    for a, b in zip(leaves(split), leaves(whole)):
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-8)
