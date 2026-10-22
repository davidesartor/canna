"""MLP and MMDiT: the scanned block stack, its identity at init, and autodiff through it."""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from canna.networks import MLP, MMDiT, MMDiTBlock

from ._helpers import close, differs, key, perturbed, rand

D, N, M, HEADS = 8, 4, 6, 2


def mlp(blocks=2, hidden=D, expand=2, seed=0):
    return MLP(hidden, blocks, expand, key=key(seed))


def mmdit(blocks=2, hidden=D, heads=HEADS, expand=2, seed=0, one_way=False):
    return MMDiT(hidden, heads, blocks, expand, one_way=one_way, key=key(seed))


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


@pytest.mark.parametrize("blocks", [1, 3])
def test_mmdit_cache_holds_every_blocks_y_keys_and_values(blocks):
    k, v = mmdit(blocks, one_way=True).encode(rand((M, D), 2), rand((D,), 3))
    assert k.shape == v.shape == (blocks, M, HEADS, D // HEADS)


def test_one_way_last_block_has_no_y_or_sink_update():
    net = MMDiT(D, HEADS, 3, sinks=True, one_way=True, key=key())
    assert net.blocks.mlp_y is not None and net.blocks.attention.out_proj_s is not None
    last = net.last
    assert last.mod_y_mlp is None and last.mlp_y is None
    assert last.mod_s_mlp is None and last.mlp_s is None
    assert last.attention.out_proj_y is None and last.attention.out_proj_s is None
    assert last.mod_s_attn is not None


def test_joint_and_one_way_refuse_each_others_calls():
    c, x, y = rand((D,), 3), rand((N, D), 1), rand((M, D), 2)
    with pytest.raises(AssertionError, match="one_way=True"):
        mmdit().encode(y, c)
    with pytest.raises(AssertionError, match="built one-way"):
        mmdit(one_way=True)(x, y, c)


def test_one_block_one_way_x_is_the_joint_x():
    """in the first block y is still its input, so x reads the same keys and values either way"""
    block, c = perturbed(MMDiTBlock(D, HEADS, 2, key=key())), rand((D,), 3)
    x, y = rand((N, D), 1), rand((M, D), 2)
    close(block.decode(x, c, block.encode(y, c)[1])[0], block(x, y, c)[0])


def test_one_way_y_never_reads_x_but_x_reads_y_once_perturbed():
    joint, c = perturbed(mmdit(blocks=3)), rand((D,), 3)
    x, y = rand((N, D), 1), rand((M, D), 2)
    # the joint net's weights, its last block stripped of the y update
    params, static = eqx.partition(joint.blocks, eqx.is_array)
    first = eqx.combine(jax.tree.map(lambda a: a[:-1], params), static)
    last = eqx.combine(jax.tree.map(lambda a: a[-1], params), static)
    last = eqx.tree_at(
        lambda b: (b.mod_y_mlp, b.mlp_y, b.attention.out_proj_y), last, (None,) * 3
    )
    net = eqx.tree_at(
        lambda n: (n.blocks, n.last), joint, (first, last), is_leaf=lambda a: a is None
    )
    cache = net.encode(y, c)
    differs(net.decode(x, c, cache), net.decode(x, c, net.encode(rand((M, D), 8), c)))
    # past the first block the joint y has read x, the one-way y has not
    differs(net.decode(x, c, cache), joint(x, y, c)[0])


def test_one_way_with_rope_matches_the_joint_rope_in_one_block():
    block, c = perturbed(MMDiTBlock(D, HEADS, 2, key=key())), rand((D,), 3)
    x, y = rand((N, D), 1), rand((M, D), 2)
    angles = rand((N + M, D // HEADS // 2), 4)
    rope = (jnp.cos(angles), jnp.sin(angles))
    rope_x, rope_y = (tuple(a[:N] for a in rope), tuple(a[N:] for a in rope))
    kv = block.encode(y, c, rope_y)[1]
    close(block.decode(x, c, kv, rope_x)[0], block(x, y, c, rope)[0])


def test_sinks_are_not_cached_and_change_x_once_perturbed():
    net = perturbed(MMDiT(D, HEADS, 3, sinks=True, one_way=True, key=key()))
    x, y, c, s = rand((N, D), 1), rand((M, D), 2), rand((D,), 3), rand((3, D), 4)
    cache = net.encode(y, c)
    assert cache[0].shape == (3, M, HEADS, D // HEADS)
    out = net.decode(x, c, cache, s=s)
    assert out.shape == (N, D)
    differs(out, net.decode(x, c, cache))
    differs(out, net.decode(x, c, cache, s=rand((3, D), 5)))


def test_rope_keeps_decode_relative():
    """GTA turns values back by the query's angle; the unpositioned sinks are left out, as they read absolute angles"""
    net = perturbed(MMDiT(D, HEADS, 2, one_way=True, key=key()))
    x, y, c = rand((N, D), 1), rand((M, D), 2), rand((D,), 3)
    periods = 2.0 + jnp.arange(D // HEADS // 2)
    rope = lambda p: (
        jnp.cos(2 * jnp.pi * p[:, None] / periods),
        jnp.sin(2 * jnp.pi * p[:, None] / periods),
    )
    run = lambda shift: net.decode(
        x,
        c,
        net.encode(y, c, rope(jnp.arange(M) * 1.7 + shift)),
        rope(jnp.arange(N) * 0.9 + shift),
    )
    close(run(5.3), run(0.0), atol=1e-4)


def test_bf16_attention_keeps_the_stream_dtype_and_caches_bf16():
    # one head of 8: cuDNN flash attention, taken on a gpu, wants head dims in multiples of 8
    build = lambda one_way, **kw: perturbed(
        MMDiT(D, 1, 2, sinks=one_way, one_way=one_way, key=key(), **kw)
    )
    plain, half = build(True), build(True, attention_dtype=jnp.bfloat16)
    x, y, c, s = rand((N, D), 1), rand((M, D), 2), rand((D,), 3), rand((3, D), 4)
    cache = half.encode(y, c)
    assert all(a.dtype == jnp.bfloat16 for a in cache)
    out = half.decode(x, c, cache, s=s)
    assert out.dtype == x.dtype
    close(out, plain.decode(x, c, plain.encode(y, c), s=s), atol=5e-2)
    plain, half = build(False), build(False, attention_dtype=jnp.bfloat16)
    joint = half(x, y, c)[0]
    assert joint.dtype == x.dtype
    close(joint, plain(x, y, c)[0], atol=5e-2)


def test_sinks_need_a_sink_stream():
    net, c = mmdit(one_way=True), rand((D,), 3)
    cache = net.encode(rand((M, D), 2), c)
    with pytest.raises(AssertionError, match="sink stream"):
        net.decode(rand((N, D), 1), c, cache, s=rand((3, D), 4))


def test_mmdit_rejects_head_counts_that_do_not_divide_the_dim():
    with pytest.raises(AssertionError, match="divisible by num_heads"):
        MMDiT(9, 2, 1, key=key())
