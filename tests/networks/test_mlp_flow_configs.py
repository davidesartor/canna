"""MLPFlow under non-default constructor args."""

import jax
import jax.numpy as jnp
from flax import nnx


from ._helpers import mlpflow, rand


def test_mlpflow_zero_blocks():
    net = mlpflow(blocks=0)
    dx, xt, yt = net(rand((3,), 1), rand((5,), 2), jnp.asarray(0.3))
    assert dx.shape == (3,) and xt.shape == (3,) and yt.shape == (5,)


def test_mlpflow_unit_dims():
    net = mlpflow(x_dim=1, y_dim=1, hidden=4, blocks=1)
    dx, xt, yt = net(rand((1,), 1), rand((1,), 2), jnp.asarray(0.3))
    assert dx.shape == (1,) and xt.shape == (1,) and yt.shape == (1,)


def test_mlpflow_odd_hidden_dim():
    net = mlpflow(hidden=15)
    dx, _, _ = net(rand((3,), 1), rand((5,), 2), jnp.asarray(0.3))
    assert dx.shape == (3,)


def test_mlpflow_blocks_are_stacked_along_a_leading_axis():
    net = mlpflow(blocks=3)
    assert net.blocks.mlp.linear1.kernel.value.shape[0] == 3


def test_mlpflow_scanned_blocks_are_not_all_the_same_block():
    net = mlpflow(blocks=3)
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


def test_mlpflow_velocity_and_x_target_share_one_projection():
    net = mlpflow(x_dim=3)
    assert net.x_unembed.linear2.kernel.value.shape[-1] == 6


def test_mlpflow_single_block():
    net = mlpflow(blocks=1)
    dx, xt, yt = net(rand((3,), 1), rand((5,), 2), jnp.asarray(0.3))
    assert dx.shape == (3,) and xt.shape == (3,) and yt.shape == (5,)


def test_mlpflow_hidden_dim_one():
    net = mlpflow(hidden=1)
    dx, _, _ = net(rand((3,), 1), rand((5,), 2), jnp.asarray(0.3))
    assert bool(jnp.all(jnp.isfinite(dx)))
