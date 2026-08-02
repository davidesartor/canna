"""MLP and MMDiT: the scanned block stack, its identity at init, and autodiff through it."""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from canna.networks import MLP, MMDiT

from ._helpers import close, differs, key, perturbed, rand

D, N, M, HEADS = 8, 4, 6, 2


def mlp(blocks=2, hidden=D, expand=2, seed=0):
    return MLP(hidden, blocks, expand, key=key(seed))


def mmdit(blocks=2, hidden=D, heads=HEADS, expand=2, seed=0):
    return MMDiT(hidden, heads, blocks, expand, key=key(seed))


@pytest.mark.parametrize("blocks", [0, 1, 3])
def test_mlp_preserves_shape(blocks):
    # MLP carries no token axis: x and c are both plain "... D"
    assert mlp(blocks)(rand((D,), 1), rand((D,), 2)).shape == (D,)


@pytest.mark.parametrize("blocks", [0, 1, 3])
def test_mmdit_preserves_both_stream_shapes(blocks):
    x, y = mmdit(blocks)(rand((N, D), 1), rand((M, D), 2), rand((D,), 3))
    assert x.shape == (N, D) and y.shape == (M, D)


def test_mlp_takes_one_example_and_vmaps():
    out = jax.vmap(mlp())(rand((5, D), 1), rand((5, D), 2))
    assert out.shape == (5, D)


def test_mmdit_takes_one_example_and_vmaps():
    x, y = jax.vmap(mmdit())(rand((5, N, D), 1), rand((5, M, D), 2), rand((5, D), 3))
    assert x.shape == (5, N, D) and y.shape == (5, M, D)


def test_mlp_is_the_identity_at_init():
    """Modulation is zero-init, so every gated residual branch starts switched off."""
    x = rand((D,), 1)
    close(mlp()(x, rand((D,), 2)), x, atol=0.0)


def test_mmdit_is_the_identity_at_init():
    x, y = rand((N, D), 1), rand((M, D), 2)
    out_x, out_y = mmdit()(x, y, rand((D,), 3))
    close(out_x, x, atol=0.0)
    close(out_y, y, atol=0.0)


def test_mlp_conditioning_reaches_the_output_once_perturbed():
    net = perturbed(mlp())
    x = rand((D,), 1)
    differs(net(x, rand((D,), 2)), net(x, rand((D,), 9)))


def test_mmdit_conditioning_reaches_both_streams_once_perturbed():
    net = perturbed(mmdit())
    x, y = rand((N, D), 1), rand((M, D), 2)
    a_x, a_y = net(x, y, rand((D,), 3))
    b_x, b_y = net(x, y, rand((D,), 9))
    differs(a_x, b_x)
    differs(a_y, b_y)


def test_mmdit_streams_attend_jointly_once_perturbed():
    """the y stream must move when only x changes, and vice versa"""
    net = perturbed(mmdit())
    y, c = rand((M, D), 2), rand((D,), 3)
    differs(net(rand((N, D), 1), y, c)[1], net(rand((N, D), 8), y, c)[1])
    x = rand((N, D), 1)
    differs(net(x, rand((M, D), 2), c)[0], net(x, rand((M, D), 8), c)[0])


def test_mlp_is_differentiable_in_x():
    """the backbone runs lax.scan over partitioned block params, which must stay differentiable"""
    net, c = perturbed(mlp()), rand((D,), 2)
    g = jax.jacobian(lambda x: net(x, c))(rand((D,), 1))
    assert g.shape == (D, D) and bool(jnp.all(jnp.isfinite(g)))
    assert not bool(jnp.all(g == 0.0))


def test_mmdit_is_differentiable_in_x():
    net, y, c = perturbed(mmdit()), rand((M, D), 2), rand((D,), 3)
    g = jax.jacobian(lambda x: net(x, y, c)[0])(rand((N, D), 1))
    assert g.shape == (N, D, N, D) and bool(jnp.all(jnp.isfinite(g)))
    assert not bool(jnp.all(g == 0.0))


@pytest.mark.parametrize("backbone", [mlp, mmdit])
def test_blocks_are_stacked_along_a_leading_axis(backbone):
    params = eqx.filter(backbone(blocks=3).blocks, eqx.is_inexact_array)
    assert all(p.shape[0] == 3 for p in jax.tree.leaves(params))


@pytest.mark.parametrize("backbone", [mlp, mmdit])
def test_scanned_blocks_are_not_all_the_same_block(backbone):
    params = eqx.filter(backbone(blocks=3).blocks, eqx.is_inexact_array)
    # constant-init params (zero gates, unit norms) repeat across blocks by design
    distinct = jax.tree.map(
        lambda p: jnp.all(p == p.reshape(-1)[0])
        | (jnp.any(p[0] != p[1]) & jnp.any(p[1] != p[2])),
        params,
    )
    assert all(jax.tree.leaves(distinct))
    varying = jax.tree.map(lambda p: jnp.any(p[0] != p[1]), params)
    assert any(jax.tree.leaves(varying))


def test_mmdit_x_stream_is_permutation_equivariant():
    net = perturbed(mmdit())
    x, y, c = rand((N, D), 1), rand((M, D), 2), rand((D,), 3)
    perm = jnp.asarray([3, 1, 0, 2])
    out_x, out_y = net(x, y, c)
    perm_x, perm_y = net(x[perm], y, c)
    close(perm_x, out_x[perm])
    close(perm_y, out_y)


def test_mmdit_rejects_head_counts_that_do_not_divide_the_dim():
    with pytest.raises(AssertionError, match="divisible by num_heads"):
        MMDiT(9, 2, 1, key=key())
