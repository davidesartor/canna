"""Fixtures shared by the lisa tests."""

from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from canna.lisa import config
from canna.lisa.chart import sample_flow
from canna.lisa.network import LisaFlow
from canna.lisa.problem import LisaGB
from canna.lisa.train import train_sample


@pytest.fixture(scope="session")
def small_flow():
    """The narrow-band problem with a deliberately tiny network: the sampler is under test."""
    problem = LisaGB(**config.load("narrow-B").problem)
    sample = eqx.filter_jit(partial(train_sample, problem))(jr.key(0))
    flow = LisaFlow(
        problem,
        hidden_dim=32,
        num_heads=2,
        num_blocks=1,
        patch=16,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        key=jr.key(1),
    )
    # a fresh MMDiT is zero-initialised into the identity transport; perturb it so the
    # flow actually depends on x, t and y
    params, static = jax.tree_util.tree_flatten(flow)
    keys = jr.split(jr.key(2), len(params))
    params = [
        p + 0.05 * jr.normal(k, p.shape, p.dtype)
        if isinstance(p, jax.Array) and jnp.issubdtype(p.dtype, jnp.floating)
        else p
        for p, k in zip(params, keys)
    ]
    flow = jax.tree_util.tree_unflatten(static, params)
    draw = eqx.filter_jit(jax.vmap(partial(sample_flow, problem), in_axes=(0, None)))
    u0 = draw(jr.split(jr.key(3), 8), sample.f)
    return problem, flow, u0, sample.y, sample.f
