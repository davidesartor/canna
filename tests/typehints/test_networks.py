"""Shape-signature contracts for the flow networks (MLPFlow, MMDiTFlow, SinusoidalEmbed)."""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from canna.networks.utils import SinusoidalEmbed, FeedForward
from canna.networks.mlp import MLPFlow
from canna.networks.mmdit import MMDiTFlow


@pytest.mark.parametrize("batch_shape", [(), (4,), (2, 3)])
def test_sinusoidal_embed_appends_dim(batch_shape):
    dim = 8
    embed = SinusoidalEmbed(dim, rngs=nnx.Rngs(0))
    t = jax.random.normal(jax.random.key(0), batch_shape)
    assert embed(t).shape == batch_shape + (dim,)


@pytest.mark.parametrize("batch_shape", [(), (5,)])
def test_feedforward_maps_in_to_out(batch_shape):
    ff = FeedForward(3, 16, 7, rngs=nnx.Rngs(0))
    x = jax.random.normal(jax.random.key(0), batch_shape + (3,))
    assert ff(x).shape == batch_shape + (7,)


@pytest.mark.parametrize("batch_shape", [(), (4,)])
def test_mlpflow_returns_dx_and_targets(batch_shape):
    X, Y = 3, 5
    flow = MLPFlow((X,), (Y,), hidden_dim=16, num_blocks=1, rngs=nnx.Rngs(0))
    x = jax.random.normal(jax.random.key(0), batch_shape + (X,))
    y = jax.random.normal(jax.random.key(1), batch_shape + (Y,))
    t = jax.random.uniform(jax.random.key(2), batch_shape)
    dx, x_target, y_target = flow(x, y, t)
    assert dx.shape == batch_shape + (X,)
    assert x_target.shape == batch_shape + (X,)
    assert y_target.shape == batch_shape + (Y,)


@pytest.mark.parametrize("batch_shape", [(), (2,)])
def test_mmditflow_returns_dx_and_targets(batch_shape):
    N, F, H, W, C = 4, 3, 8, 8, 2
    flow = MMDiTFlow(
        (N, F),
        (H, W, C),
        hidden_dim=16,
        num_heads=2,
        num_blocks=1,
        patch_stages=1,
        rngs=nnx.Rngs(0),
    )
    x = jax.random.normal(jax.random.key(0), batch_shape + (N, F))
    y = jax.random.normal(jax.random.key(1), batch_shape + (H, W, C))
    t = jax.random.uniform(jax.random.key(2), batch_shape)
    dx, x_target, y_target = flow(x, y, t)
    assert dx.shape == batch_shape + (N, F)
    assert x_target.shape == batch_shape + (N, F)
    assert y_target.shape == batch_shape + (H, W, C)
