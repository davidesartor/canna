"""Builders and comparisons shared by the network test files."""

import jax
import numpy as np
from flax import nnx

from canna.networks.mlp import MLPFlow
from canna.networks.mmdit import MMDiTFlow


def rand(shape, seed: int = 0):
    return jax.random.normal(jax.random.key(seed), shape)


def rngs(seed: int = 0):
    return nnx.Rngs(seed)


def close(a, b, atol=1e-4):
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=atol, rtol=1e-4)


def differs(a, b, atol=1e-5):
    assert not np.allclose(np.asarray(a), np.asarray(b), atol=atol)


def mmditflow(
    x_dim=2,
    y_dim=3,
    hidden=16,
    heads=2,
    blocks=2,
    expand=2,
    stages=2,
    seed=0,
    sources=4,
    height=8,
    width=8,
):
    return MMDiTFlow(
        (sources, x_dim),
        (height, width, y_dim),
        hidden,
        heads,
        blocks,
        expand,
        stages,
        rngs=rngs(seed),
    )


def mlpflow(x_dim=3, y_dim=5, hidden=16, blocks=2, expand=2, seed=0):
    return MLPFlow((x_dim,), (y_dim,), hidden, blocks, expand, rngs=rngs(seed))


def perturbed(module, seed: int = 7, scale: float = 0.3):
    """Move every param off init: zero-init gates make a fresh network ignore its conditioning."""
    state = nnx.state(module, nnx.Param)
    leaves, treedef = jax.tree.flatten(state)
    keys = jax.random.split(jax.random.key(seed), len(leaves))
    noised = [
        p + scale * jax.random.normal(k, p.shape, p.dtype) for p, k in zip(leaves, keys)
    ]
    nnx.update(module, jax.tree.unflatten(treedef, noised))
    return module
