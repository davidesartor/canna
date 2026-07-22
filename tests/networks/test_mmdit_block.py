"""MMDiTBlock: one modulated joint-attention layer over both streams."""

import jax.numpy as jnp
import pytest

from canna.networks.mmdit import MMDiTBlock

from ._helpers import close, differs, rand, rngs


# ---------------------------------------------------------------- MMDiTBlock


def test_mmditblock_preserves_both_stream_shapes():
    blk = MMDiTBlock(8, 2, 2, rngs=rngs())
    ox, oy = blk(rand((5, 8), 1), rand((3, 8), 2), rand((1, 8), 3))
    assert ox.shape == (5, 8) and oy.shape == (3, 8)


def test_mmditblock_batch_ranks():
    blk = MMDiTBlock(8, 2, 2, rngs=rngs())
    for lead in [(), (4,), (2, 4)]:
        ox, oy = blk(
            rand(lead + (5, 8), 1), rand(lead + (3, 8), 2), rand(lead + (1, 8), 3)
        )
        assert ox.shape == lead + (5, 8)
        assert oy.shape == lead + (3, 8)


def test_mmditblock_batch_independent():
    blk = MMDiTBlock(8, 2, 2, rngs=rngs())
    x, y, c = rand((4, 5, 8), 1), rand((4, 3, 8), 2), rand((4, 1, 8), 3)
    ox, oy = blk(x, y, c)
    parts = [blk(x[i], y[i], c[i]) for i in range(4)]
    close(ox, jnp.stack([p[0] for p in parts]))
    close(oy, jnp.stack([p[1] for p in parts]))


def test_mmditblock_identity_at_init_is_bit_exact():
    blk = MMDiTBlock(8, 2, 2, rngs=rngs())
    x, y, c = rand((5, 8), 1), rand((3, 8), 2), rand((1, 8), 3)
    ox, oy = blk(x, y, c)
    close(ox, x, atol=0.0)
    close(oy, y, atol=0.0)


def test_mmditblock_depends_on_c_with_perturbed_params(perturbed_block):
    blk = perturbed_block
    x, y = rand((5, 8), 1), rand((3, 8), 2)
    differs(blk(x, y, rand((1, 8), 3))[0], blk(x, y, rand((1, 8), 11))[0])


def test_mmditblock_x_permutation_equivariant_with_perturbed_params(perturbed_block):
    blk = perturbed_block
    x, y, c = rand((5, 8), 1), rand((3, 8), 2), rand((1, 8), 3)
    perm = jnp.asarray([4, 3, 0, 2, 1])
    ox, oy = blk(x, y, c)
    px, py = blk(x[perm], y, c)
    close(px, ox[perm])
    close(py, oy)


def test_mmditblock_streams_mix_with_perturbed_params(perturbed_block):
    blk = perturbed_block
    x, c = rand((5, 8), 1), rand((1, 8), 3)
    differs(blk(x, rand((3, 8), 2), c)[0], blk(x, rand((3, 8), 9), c)[0])


def test_mmditblock_accepts_per_token_conditioning(perturbed_block):
    blk = perturbed_block
    ox, oy = blk(rand((5, 8), 1), rand((5, 8), 2), rand((5, 8), 3))
    assert ox.shape == (5, 8) and oy.shape == (5, 8)


def test_mmditblock_rejects_conditioning_with_a_mismatched_token_count():
    blk = MMDiTBlock(8, 2, 2, rngs=rngs())
    with pytest.raises(Exception):
        blk(rand((5, 8), 1), rand((3, 8), 2), rand((2, 8), 3))


def test_mmditblock_uses_four_independent_modulations(perturbed_block):
    blk = perturbed_block
    x, y, c = rand((5, 8), 1), rand((3, 8), 2), rand((1, 8), 3)
    differs(blk.mod_x_attn(x, c)[1], blk.mod_y_attn(y, c)[1])
    differs(blk.mod_x_attn(x, c)[1], blk.mod_x_mlp(x, c)[1])


def test_mmditblock_expand_one():
    blk = MMDiTBlock(8, 2, 1, rngs=rngs())
    ox, oy = blk(rand((5, 8), 1), rand((3, 8), 2), rand((1, 8), 3))
    assert ox.shape == (5, 8) and oy.shape == (3, 8)
