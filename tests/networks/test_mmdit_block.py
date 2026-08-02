"""MMDiTBlock: one modulated joint-attention layer over both streams."""

import jax
import jax.numpy as jnp
import pytest

from canna.networks import MMDiTBlock

from ._helpers import close, differs, key, rand

# ---------------------------------------------------------------- MMDiTBlock


def test_mmditblock_preserves_both_stream_shapes():
    blk = MMDiTBlock(8, 2, 2, key=key())
    ox, oy = blk(rand((5, 8), 1), rand((3, 8), 2), rand((8,), 3))
    assert ox.shape == (5, 8) and oy.shape == (3, 8)


def test_mmditblock_takes_one_example_and_vmaps():
    blk = MMDiTBlock(8, 2, 2, key=key())
    ox, oy = jax.vmap(blk)(rand((4, 5, 8), 1), rand((4, 3, 8), 2), rand((4, 8), 3))
    assert ox.shape == (4, 5, 8)
    assert oy.shape == (4, 3, 8)


def test_mmditblock_batch_independent():
    blk = MMDiTBlock(8, 2, 2, key=key())
    x, y, c = rand((4, 5, 8), 1), rand((4, 3, 8), 2), rand((4, 8), 3)
    ox, oy = jax.vmap(blk)(x, y, c)
    parts = [blk(x[i], y[i], c[i]) for i in range(4)]
    close(ox, jnp.stack([p[0] for p in parts]))
    close(oy, jnp.stack([p[1] for p in parts]))


def test_mmditblock_identity_at_init_is_bit_exact():
    blk = MMDiTBlock(8, 2, 2, key=key())
    x, y, c = rand((5, 8), 1), rand((3, 8), 2), rand((8,), 3)
    ox, oy = blk(x, y, c)
    close(ox, x, atol=0.0)
    close(oy, y, atol=0.0)


def test_mmditblock_depends_on_c_with_perturbed_params(perturbed_block):
    blk = perturbed_block
    x, y = rand((5, 8), 1), rand((3, 8), 2)
    differs(blk(x, y, rand((8,), 3))[0], blk(x, y, rand((8,), 11))[0])


def test_mmditblock_x_permutation_equivariant_with_perturbed_params(perturbed_block):
    blk = perturbed_block
    x, y, c = rand((5, 8), 1), rand((3, 8), 2), rand((8,), 3)
    perm = jnp.asarray([4, 3, 0, 2, 1])
    ox, oy = blk(x, y, c)
    px, py = blk(x[perm], y, c)
    close(px, ox[perm])
    close(py, oy)


def test_mmditblock_streams_mix_with_perturbed_params(perturbed_block):
    blk = perturbed_block
    x, c = rand((5, 8), 1), rand((8,), 3)
    differs(blk(x, rand((3, 8), 2), c)[0], blk(x, rand((3, 8), 9), c)[0])


def test_mmditblock_rejects_per_token_conditioning():
    blk = MMDiTBlock(8, 2, 2, key=key())
    with pytest.raises(Exception):
        blk(rand((5, 8), 1), rand((3, 8), 2), rand((5, 8), 3))


def test_mmditblock_rejects_conditioning_with_a_mismatched_dim():
    blk = MMDiTBlock(8, 2, 2, key=key())
    with pytest.raises(Exception):
        blk(rand((5, 8), 1), rand((3, 8), 2), rand((2,), 3))


def test_mmditblock_uses_four_independent_modulations(perturbed_block):
    blk = perturbed_block
    x, y, c = rand((5, 8), 1), rand((3, 8), 2), rand((8,), 3)
    differs(blk.mod_x_attn(x, c)[1], blk.mod_y_attn(y, c)[1])
    differs(blk.mod_x_attn(x, c)[1], blk.mod_x_mlp(x, c)[1])


def test_mmditblock_expand_one():
    blk = MMDiTBlock(8, 2, 1, key=key())
    ox, oy = blk(rand((5, 8), 1), rand((3, 8), 2), rand((8,), 3))
    assert ox.shape == (5, 8) and oy.shape == (3, 8)
