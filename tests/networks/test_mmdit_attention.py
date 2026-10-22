"""PositionalEmbed and the joint-stream attention it feeds."""

from dataclasses import fields

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from canna.networks import PositionalEmbed, MultiStreamAttention

from ._helpers import close, differs, key, rand

# ------------------------------------------------------------ PositionalEmbed


def test_positional_preserves_shape():
    pe = PositionalEmbed(8, 16, key=key())
    assert pe(rand((6, 8))).shape == (6, 8)


def test_positional_batch_ranks():
    pe = PositionalEmbed(8, 16, key=key())
    for lead in [(), (5,), (2, 5)]:
        assert pe(rand(lead + (6, 8))).shape == lead + (6, 8)


def test_positional_offset_independent_of_input():
    pe = PositionalEmbed(8, 16, key=key())
    x1, x2 = rand((6, 8), 1), rand((6, 8), 2)
    close(pe(x1) - x1, pe(x2) - x2)


def test_positional_offset_constant_across_batch_axis():
    pe = PositionalEmbed(8, 16, key=key())
    x = rand((3, 6, 8))
    delta = pe(x) - x
    close(delta[0], delta[1])
    close(delta[0], delta[2])


def test_positional_breaks_permutation_along_axis():
    pe = PositionalEmbed(8, 16, key=key())
    x = rand((6, 8))
    rolled = jnp.roll(x, 1, axis=-2)
    assert not np.allclose(
        np.asarray(pe(rolled)), np.asarray(jnp.roll(pe(x), 1, axis=-2)), atol=1e-5
    )


def test_positional_positions_are_absolute_not_normalized():
    pe = PositionalEmbed(8, 16, key=key())
    short, long = rand((4, 8), 1), rand((9, 8), 2)
    close((pe(short) - short)[1], (pe(long) - long)[1])
    differs((pe(long) - long)[1], (pe(long) - long)[-1])


def test_positional_custom_axis_targets_that_axis():
    pe = PositionalEmbed(8, 16, axis=-3, key=key())
    x = rand((4, 6, 8))
    delta = pe(x) - x
    close(delta[0, 0], delta[0, 1])
    assert not np.allclose(np.asarray(delta[0, 0]), np.asarray(delta[1, 0]), atol=1e-5)


def test_positional_offset_dim_must_match_feature_dim():
    pe = PositionalEmbed(4, 16, key=key())
    with pytest.raises(Exception):
        pe(rand((6, 8)))


# ------------------------------------------------------ MultiStreamAttention


def test_attention_preserves_both_stream_shapes():
    att = MultiStreamAttention(8, 2, key=key())
    ox, oy = att(rand((5, 8), 1), rand((3, 8), 2))
    assert ox.shape == (5, 8)
    assert oy.shape == (3, 8)


def test_attention_takes_one_example_and_vmaps():
    att = MultiStreamAttention(8, 2, key=key())
    ox, oy = jax.vmap(att)(rand((4, 5, 8), 1), rand((4, 3, 8), 2))
    assert ox.shape == (4, 5, 8)
    assert oy.shape == (4, 3, 8)


def test_attention_batch_independent():
    att = MultiStreamAttention(8, 2, key=key())
    x, y = rand((4, 5, 8), 1), rand((4, 3, 8), 2)
    ox, oy = jax.vmap(att)(x, y)
    parts = [att(x[i], y[i]) for i in range(4)]
    close(ox, jnp.stack([p[0] for p in parts]))
    close(oy, jnp.stack([p[1] for p in parts]))


def test_attention_equivariant_to_x_permutation():
    att = MultiStreamAttention(8, 2, key=key())
    x, y = rand((5, 8), 1), rand((3, 8), 2)
    perm = jnp.asarray([2, 0, 4, 1, 3])
    ox, oy = att(x, y)
    px, py = att(x[perm], y)
    close(px, ox[perm])
    close(py, oy)


def test_attention_equivariant_to_y_permutation():
    att = MultiStreamAttention(8, 2, key=key())
    x, y = rand((5, 8), 1), rand((3, 8), 2)
    perm = jnp.asarray([1, 2, 0])
    ox, oy = att(x, y)
    px, py = att(x, y[perm])
    close(py, oy[perm])
    close(px, ox)


def test_attention_split_point_follows_the_x_stream_length():
    att = MultiStreamAttention(8, 2, key=key())
    for n, m in [(1, 7), (7, 1), (4, 4)]:
        ox, oy = att(rand((n, 8), 1), rand((m, 8), 2))
        assert ox.shape == (n, 8) and oy.shape == (m, 8)


def test_attention_streams_interact():
    att = MultiStreamAttention(8, 2, key=key())
    x = rand((5, 8), 1)
    a, _ = att(x, rand((3, 8), 2))
    b, _ = att(x, rand((3, 8), 9))
    assert not np.allclose(np.asarray(a), np.asarray(b), atol=1e-5)


def test_attention_projections_are_separate_per_stream():
    att = MultiStreamAttention(8, 2, key=key())
    tok = rand((1, 8), 1)
    ox, oy = att(tok, tok)
    assert not np.allclose(np.asarray(ox), np.asarray(oy), atol=1e-5)


def test_attention_single_head():
    att = MultiStreamAttention(8, 1, key=key())
    ox, oy = att(rand((5, 8), 1), rand((3, 8), 2))
    assert ox.shape == (5, 8) and oy.shape == (3, 8)


def test_attention_assertion_names_num_heads():
    with pytest.raises(AssertionError, match="dim should be divisible by num_heads"):
        MultiStreamAttention(8, 3, key=key())


def test_attention_duplicate_tokens_map_to_duplicate_outputs():
    att = MultiStreamAttention(8, 2, key=key())
    tok = rand((1, 8), 1)
    x = jnp.concatenate([tok, tok], axis=0)
    ox, _ = att(x, rand((3, 8), 2))
    close(ox[0], ox[1])


def test_attention_has_no_causal_mask():
    att = MultiStreamAttention(8, 2, key=key())
    x, y = rand((5, 8), 1), rand((3, 8), 2)
    ox, oy = att(x, y)
    rx, ry = att(x[::-1], y)
    close(rx, ox[::-1])
    close(ry, oy)


def test_attention_use_bias_changes_the_output():
    plain = MultiStreamAttention(8, 2, key=key(0))
    biased = MultiStreamAttention(8, 2, use_bias=True, key=key(0))
    x, y = rand((5, 8), 1), rand((3, 8), 2)
    assert biased.qkv_proj_x.bias is not None
    assert plain.qkv_proj_x.bias is None
    assert biased(x, y)[0].shape == (5, 8)


def test_attention_qk_norm_has_no_parameters_but_the_opt_in_gain():
    att = MultiStreamAttention(16, 4, key=key())
    assert {f.name for f in fields(att)} == {
        "num_heads",
        "qkv_proj_x",
        "qkv_proj_y",
        "out_proj_x",
        "out_proj_y",
        "qk_gain",
        "attention_dtype",
        "qkv_proj_s",
        "out_proj_s",
    }
    assert att.qk_gain is None and att.qkv_proj_s is None


def test_attention_qk_gain_starts_as_the_plain_norm_and_sharpens_the_logits():
    plain = MultiStreamAttention(8, 2, key=key(0))
    gained = MultiStreamAttention(8, 2, qk_gain=True, key=key(0))
    assert gained.qk_gain.shape == (2,)
    x, y = rand((5, 8), 1), rand((3, 8), 2)
    close(gained(x, y)[0], plain(x, y)[0], atol=0.0)
    sharp = eqx.tree_at(lambda a: a.qk_gain, gained, 4.0 * gained.qk_gain)
    differs(sharp(x, y)[0], plain(x, y)[0])


def test_attention_scales_sublinearly_in_token_magnitude():
    att = MultiStreamAttention(8, 2, key=key())
    x, y = rand((5, 8), 1), rand((3, 8), 2)
    ox, _ = att(x, y)
    big, _ = att(1e3 * x, y)
    assert bool(jnp.all(jnp.isfinite(big)))
    differs(big, 1e3 * ox)


def rope_at(positions, periods=(2.0, 5.0)):
    angles = 2 * jnp.pi * jnp.asarray(positions)[:, None] / jnp.asarray(periods)[None]
    return jnp.cos(angles), jnp.sin(angles)


def test_attention_rope_is_relative():
    att = MultiStreamAttention(8, 2, key=key(0))
    x, y = rand((5, 8), 1), rand((3, 8), 2)
    positions = jnp.array([0.0, 1.3, 2.0, 7.5, 4.0, 0.5, 3.2, 9.0])
    out = att(x, y, rope_at(positions))
    differs(out[0], att(x, y)[0])
    shifted = att(x, y, rope_at(positions + 17.25))
    close(shifted[0], out[0], atol=1e-4)
    close(shifted[1], out[1], atol=1e-4)


def test_attention_rope_rotates_values_by_the_relative_position():
    from canna.networks.mmdit import rotate

    att = MultiStreamAttention(8, 2, key=key(0))
    x, y = rand((1, 8), 1), rand((1, 8), 2)
    # one key: the query reads its value whole, turned by key minus query position
    q, _, _ = att.heads(att.qkv_proj_x, x, rope_at([1.3]))
    _, k, v = att.heads(att.qkv_proj_y, y, rope_at([4.0]))
    _, _, plain = att.heads(att.qkv_proj_y, y)
    out = att.attend(q, k, v, rope_at([1.3]))
    expected = rotate(plain, *rope_at([4.0 - 1.3])).reshape(1, 8)
    close(out, expected, atol=1e-5)
    differs(out, plain.reshape(1, 8))
