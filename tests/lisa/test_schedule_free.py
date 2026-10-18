"""canna.lisa.schedule_free: optax's schedule-free when the fixes are off, and each fix (F37)."""

import jax
import jax.numpy as jnp
import numpy as np
import optax

from canna.lisa import schedule_free as sf

TARGET = jnp.array([[1.0, -2.0], [0.5, 3.0]])


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
    ours = sf.schedule_free(
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
        lambda p: optax.EmptyState(), lambda g, s, p=None: (jax.tree.map(lambda a: 0.01 + 0 * a, g), s)
    )
    lr, lam = 0.1, 0.5
    free = sf.schedule_free(push, lambda s: lr, weight_decay=0.0)
    held = sf.schedule_free(push, lambda s: lr, weight_decay=lam)
    _, s_free = run(free, 2000)
    _, s_held = run(held, 2000)
    # without decay z walks off linearly; with it, z -> u / (lr lambda)
    assert np.allclose(s_free.z["w"], 20.0, rtol=1e-3)
    np.testing.assert_allclose(s_held.z["w"], 0.01 / (lr * lam), rtol=1e-3)


def test_x_follows_z_until_the_warm_start():
    tx = sf.schedule_free(optax.sgd(0.1), lambda s: 0.1, warm_start=10)
    for steps in (1, 5, 10):
        _, s = run(tx, steps)
        np.testing.assert_allclose(s.x["w"], s.z["w"])
    _, s = run(tx, 20)
    assert not np.allclose(s.x["w"], s.z["w"])


def test_late_iterates_weigh_more_with_r_one():
    # on a target reached early, r=1 pulls x towards the late iterates faster than r=0
    uniform = sf.schedule_free(optax.sgd(0.05), lambda s: 0.05, r=0.0)
    late = sf.schedule_free(optax.sgd(0.05), lambda s: 0.05, r=1.0)
    _, s0 = run(uniform, 200)
    _, s1 = run(late, 200)
    err = lambda s: float(jnp.linalg.norm(s.x["w"] - TARGET))
    assert err(s1) < err(s0)


def test_interpolation_runs_from_start_to_end():
    b = lambda t: float(sf.interpolation(jnp.asarray(t), 0.9, 0.965, 100, 10_000))
    assert b(1) == b(100) == np.float32(0.9)
    assert np.isclose(b(10_000), 0.965) and np.isclose(b(50_000), 0.965)
    values = [b(t) for t in (100, 300, 1000, 3000, 10_000)]
    assert all(a < c for a, c in zip(values, values[1:]))


def test_l1_weighting_is_off_until_the_warm_start_then_divides_by_the_gradient_l1():
    unit = optax.GradientTransformation(lambda p: optax.EmptyState(), lambda g, s, p=None: (g, s))
    tx = sf.schedule_free(unit, lambda s: 1.0, warm_start=3, l1_weighting=True, l1_beta=0.0)
    params = {"w": jnp.zeros((2, 2))}
    state = tx.init(params)
    g = {"w": jnp.ones((2, 2))}
    for _ in range(3):  # warm start: the reference is set, no weighting
        _, state = tx.update(g, state, params)
    z_before = state.z["w"]
    _, state = tx.update(jax.tree.map(lambda a: 4.0 * a, g), state, params)
    # with l1_beta 0 the EMA is the current |g|_1, four times the reference: step 4 * 1/4
    np.testing.assert_allclose(state.z["w"] - z_before, 1.0, rtol=1e-6)


def test_normuon_rows_keeps_each_matrix_norm_and_evens_out_its_rows():
    import argparse
    from pathlib import Path
    import yaml
    import canna.lisa as lisa
    from canna.lisa.train import TrainState, normuon_rows

    with open(Path(lisa.__file__).parent / "configs" / "XS-fd.yaml") as f:
        config = yaml.safe_load(f)
    config["network"] = dict(hidden_dim=32, num_blocks=1, num_heads=2, patch=16)
    flow = TrainState.from_config(argparse.Namespace(**config, dtype="float32")).flow
    import equinox as eqx
    params = eqx.filter(flow, eqx.is_inexact_array)
    # rows of very different sizes in every Muon matrix
    updates = jax.tree.map(
        lambda p: p * 0 + jnp.arange(1.0, 1.0 + p.shape[-2])[:, None] if p.ndim >= 2 else p * 0 + 1.0, params
    )
    tx = normuon_rows()
    out, _ = tx.update(updates, tx.init(params), params)
    u = updates.backbone.blocks.mlp_y.linear2.weight
    o = out.backbone.blocks.mlp_y.linear2.weight
    np.testing.assert_allclose(jnp.linalg.norm(o[0]), jnp.linalg.norm(u[0]), rtol=1e-4)
    rows_u, rows_o = jnp.linalg.norm(u[0], axis=-1), jnp.linalg.norm(o[0], axis=-1)
    assert float(rows_o.max() / rows_o.min()) < 1.01 < float(rows_u.max() / rows_u.min())
    # AdamW's leaves are left alone
    np.testing.assert_allclose(out.x_unembed.linear2.weight, updates.x_unembed.linear2.weight)


def test_the_optimizer_builds_and_steps_with_and_without_schedule_free():
    import argparse
    from pathlib import Path
    import yaml
    import canna.lisa as lisa
    from canna.lisa.train import TrainState
    import jax.random as jr

    with open(Path(lisa.__file__).parent / "configs" / "XS-fd.yaml") as f:
        config = yaml.safe_load(f)
    config["network"] = dict(hidden_dim=32, num_blocks=1, num_heads=2, patch=16)
    for use_sf in (True, False):
        args = argparse.Namespace(**{**config, "batch_size": 4}, dtype="float32", schedule_free=use_sf)
        state = TrainState.from_config(args)
        state, losses = state.train_epoch(4, 3, 3.0, 0.1)
        assert np.all(np.isfinite(np.asarray(losses)))


def test_decay_adamw_off_decays_the_muon_matrices_only():
    import argparse
    from pathlib import Path
    import yaml
    import equinox as eqx
    import canna.lisa as lisa
    from canna.lisa.train import TrainState, optimizer, schedule_free_state

    with open(Path(lisa.__file__).parent / "configs" / "XS-fd.yaml") as f:
        config = yaml.safe_load(f)
    config["network"] = dict(hidden_dim=32, num_blocks=1, num_heads=2, patch=16)
    config.update(warmup_steps=1, weight_decay=50.0)  # a few steps must show it
    flow = TrainState.from_config(argparse.Namespace(**config, dtype="float32")).flow
    params = eqx.filter(flow, eqx.is_inexact_array)
    zeros = jax.tree.map(jnp.zeros_like, params)
    head = lambda p: p.x_unembed.linear2.weight  # AdamW
    block = lambda p: p.backbone.blocks.mlp_y.linear2.weight  # Muon
    for use_sf in (True, False):
        for decay_adamw in (True, False):
            args = argparse.Namespace(**config, schedule_free=use_sf, decay_adamw=decay_adamw)
            tx = optimizer(args)
            p, state = params, tx.init(params)
            for _ in range(3):
                u, state = tx.update(zeros, state, p)
                p = optax.apply_updates(p, u)
            z = schedule_free_state(state).z if use_sf else p
            assert float(jnp.linalg.norm(block(z))) < 0.99 * float(jnp.linalg.norm(block(params)))
            shrunk = float(jnp.linalg.norm(head(z))) < 0.99 * float(jnp.linalg.norm(head(params)))
            assert shrunk == decay_adamw
