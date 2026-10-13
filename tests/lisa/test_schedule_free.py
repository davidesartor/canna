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
