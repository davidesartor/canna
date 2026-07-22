"""MMDiTFlow under non-default constructor args; each case builds its own network."""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from canna.networks.mmdit import MMDiTFlow

from ._helpers import close, mmditflow, perturbed, rand, rngs


def test_mmditflow_non_square_image():
    net = mmditflow(height=8, width=16)
    dx, _, yt = net(rand((4, 2), 1), rand((8, 16, 3), 2), jnp.asarray(0.3))
    assert dx.shape == (4, 2) and yt.shape == (8, 16, 3)


def test_mmditflow_zero_blocks():
    net = mmditflow(blocks=0)
    dx, xt, yt = net(rand((4, 2), 1), rand((8, 8, 3), 2), jnp.asarray(0.3))
    assert dx.shape == (4, 2) and xt.shape == (4, 2) and yt.shape == (8, 8, 3)


def test_mmditflow_single_channel_image():
    net = mmditflow(y_dim=1)
    dx, _, yt = net(rand((4, 2), 1), rand((8, 8, 1), 2), jnp.asarray(0.3))
    assert dx.shape == (4, 2) and yt.shape == (8, 8, 1)


def test_mmditflow_velocity_is_differentiable_in_x():
    y, t, x_shape = (rand((8, 8, 3), 2), jnp.asarray(0.4), (4, 2))
    graphdef, state = nnx.split(mmditflow())

    def velocity(state, z):
        return nnx.merge(graphdef, state)(z, y, t)[0]

    g = jax.jacobian(velocity, argnums=1)(state, rand(x_shape, 1))
    assert g.shape == x_shape + x_shape and bool(jnp.all(jnp.isfinite(g)))


def test_mmditflow_x_stream_has_no_positional_embedding():
    net = mmditflow()
    tok = rand((1, 2), 1)
    x = jnp.concat([tok, tok + 1e-9], axis=0)
    dx = perturbed(net)(x, rand((8, 8, 3), 2), jnp.asarray(0.3))[0]
    close(dx[0], dx[1], atol=1e-5)


def test_mmditflow_default_patch_stages_needs_hidden_divisible_by_64():
    with pytest.raises(AssertionError, match="hidden_dim must be divisible by 64"):
        MMDiTFlow((4, 2), (16, 16, 3), 16, 2, 2, rngs=rngs())


def test_mmditflow_documented_default_config_runs():
    net = MMDiTFlow((4, 2), (16, 16, 3), 64, 4, 1, rngs=rngs())
    dx, xt, yt = net(rand((4, 2), 1), rand((16, 16, 3), 2), jnp.asarray(0.3))
    assert dx.shape == (4, 2) and xt.shape == (4, 2) and yt.shape == (16, 16, 3)


@pytest.mark.parametrize("stages", [1, 2, 3])
def test_mmditflow_y_target_shape_survives_every_patch_stage_count(stages):
    net = mmditflow(hidden=64, stages=stages, height=16, width=16)
    yt = net(rand((4, 2), 1), rand((16, 16, 3), 2), jnp.asarray(0.3))[2]
    assert yt.shape == (16, 16, 3)


def test_mmditflow_image_too_small_for_the_patch_stages():
    net = mmditflow(hidden=64, stages=3)
    with pytest.raises(Exception):
        net(rand((4, 4, 3), 1), rand((4, 4, 3), 2), jnp.asarray(0.3))


def test_mmditflow_scanned_blocks_are_not_all_the_same_block():
    net = mmditflow(blocks=3)
    params = nnx.state(net.blocks, nnx.Param)
    # constant-init params (zero gates, unit norms) repeat across blocks by design
    distinct = jax.tree.map(
        lambda p: jnp.all(p == p.reshape(-1)[0])
        | (jnp.any(p[0] != p[1]) & jnp.any(p[1] != p[2])),
        params,
    )
    assert all(jax.tree.leaves(distinct))
    varying = jax.tree.map(lambda p: jnp.any(p[0] != p[1]), params)
    assert any(jax.tree.leaves(varying))
