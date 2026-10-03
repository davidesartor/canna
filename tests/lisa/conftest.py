"""Fixtures shared by the lisa tests."""

from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest
import yaml

import canna.lisa as lisa
from canna.lisa import LisaFlow, LisaGB, train_sample


@pytest.fixture(scope="module")
def small_flow():
    """The XS problem with a deliberately tiny network: the integrator is under test."""
    with open(Path(lisa.__file__).parent / "configs" / "XS.yaml") as f:
        problem = LisaGB(**yaml.safe_load(f)["problem"])
    sample = train_sample(problem, jr.key(0))
    flow = LisaFlow(
        x_shape=sample.xt.shape,
        y_shape=sample.y.shape,
        hidden_dim=32,
        num_heads=2,
        num_blocks=1,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        key=jr.key(1),
    )
    # a fresh MMDiT is zero-initialised into the identity transport; perturb it so the
    # velocity field actually depends on x, t and y
    params, static = jax.tree_util.tree_flatten(flow)
    keys = jr.split(jr.key(2), len(params))
    params = [
        p + 0.05 * jr.normal(k, p.shape, p.dtype)
        if isinstance(p, jax.Array) and jnp.issubdtype(p.dtype, jnp.floating)
        else p
        for p, k in zip(params, keys)
    ]
    flow = jax.tree_util.tree_unflatten(static, params)
    u0 = jax.vmap(problem.sample_flow, in_axes=(0, None))(jr.split(jr.key(3), 8), sample.f)
    return problem, flow, u0, sample.y, sample.f


