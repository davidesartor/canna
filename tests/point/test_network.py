"""PointFlow: shapes, batching, and which of (x, t, y) the velocity may see."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from canna.point import PointFlow

X, Y, HIDDEN = 3, 5, 8


def rand(shape, seed: int = 0):
    return jax.random.normal(jax.random.key(seed), shape)


def close(a, b, atol=1e-4):
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=atol, rtol=1e-4)


def differs(a, b, atol=1e-5):
    assert not np.allclose(np.asarray(a), np.asarray(b), atol=atol)


def flow(x_dim=X, y_dim=Y, hidden=HIDDEN, blocks=2, seed=0):
    return PointFlow((x_dim,), (y_dim,), hidden, blocks, key=jax.random.key(seed))


def perturbed(module, seed: int = 7, scale: float = 0.3):
    """Move every param off init: zero-init gates make a fresh network ignore its conditioning."""
    params, static = eqx.partition(module, eqx.is_inexact_array)
    leaves, treedef = jax.tree.flatten(params)
    keys = jax.random.split(jax.random.key(seed), len(leaves))
    noised = [
        p + scale * jax.random.normal(k, p.shape, p.dtype) for p, k in zip(leaves, keys)
    ]
    return eqx.combine(jax.tree.unflatten(treedef, noised), static)


def test_output_shape():
    dx = flow()(rand((X,), 1), jnp.asarray(0.3), rand((Y,), 2))
    assert dx.shape == (X,)


def test_takes_one_example_and_vmaps():
    dx = jax.vmap(flow())(rand((4, X), 1), rand((4,), 3), rand((4, Y), 2))
    assert dx.shape == (4, X)


def test_batch_rows_are_independent():
    net = perturbed(flow())
    x, t, y = rand((4, X), 1), rand((4,), 3), rand((4, Y), 2)
    dx = jax.vmap(net)(x, t, y)
    close(dx, jnp.stack([net(x[i], t[i], y[i]) for i in range(4)]))


def test_velocity_depends_on_x_at_init():
    """the x stream is a plain residual path, so x reaches dx even with the gates off"""
    net, t, y = flow(), jnp.asarray(0.4), rand((Y,), 2)
    differs(net(rand((X,), 1), t, y), net(rand((X,), 8), t, y))


def test_unembed_projects_to_exactly_one_head():
    assert flow().x_unembed.linear2.weight.shape[0] == X


def test_is_deterministic():
    net = flow()
    args = (rand((X,), 1), rand(()), rand((Y,), 2))
    close(net(*args), net(*args))


@pytest.mark.parametrize("t", [0.0, 1.0])
def test_finite_at_the_time_endpoints(t):
    out = flow()(rand((X,), 1), jnp.asarray(t), rand((Y,), 2))
    assert bool(jnp.all(jnp.isfinite(out)))


def test_ignores_conditioning_at_init():
    """Modulation is zero-init, so a fresh network's velocity is blind to t and y"""
    net, x = flow(), rand((X,), 1)
    a = net(x, jnp.asarray(0.1), rand((Y,), 2))
    b = net(x, jnp.asarray(0.9), 10.0 * rand((Y,), 9))
    close(a, b, atol=0.0)


def test_velocity_depends_on_t_once_perturbed():
    net, x, y = perturbed(flow()), rand((X,), 1), rand((Y,), 2)
    differs(net(x, jnp.asarray(0.1), y), net(x, jnp.asarray(0.9), y))


def test_velocity_depends_on_y_once_perturbed():
    net, x, t = perturbed(flow()), rand((X,), 1), jnp.asarray(0.4)
    differs(net(x, t, rand((Y,), 2)), net(x, t, rand((Y,), 7)))


def test_velocity_is_differentiable_in_x():
    net, t, y = perturbed(flow()), jnp.asarray(0.4), rand((Y,), 2)
    g = jax.jacobian(lambda z: net(z, t, y))(rand((X,), 1))
    assert g.shape == (X, X) and bool(jnp.all(jnp.isfinite(g)))


def test_perturbed_outputs_stay_finite():
    out = jax.vmap(perturbed(flow()))(rand((4, X), 1), rand((4,), 3), rand((4, Y), 2))
    assert bool(jnp.all(jnp.isfinite(out)))


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(blocks=0),
        dict(blocks=1),
        dict(hidden=1),
        dict(hidden=15),
        dict(x_dim=1, y_dim=1, hidden=4, blocks=1),
    ],
)
def test_odd_shapes_still_build_and_run(kwargs):
    net = flow(**kwargs)
    x_dim, y_dim = kwargs.get("x_dim", X), kwargs.get("y_dim", Y)
    dx = net(rand((x_dim,), 1), jnp.asarray(0.3), rand((y_dim,), 2))
    assert dx.shape == (x_dim,)
    assert bool(jnp.all(jnp.isfinite(dx)))
