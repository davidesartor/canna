"""MLPBlock: one modulated residual layer."""

import jax
import jax.numpy as jnp

from canna.networks import MLPBlock

from ._helpers import close, differs, key, perturbed, rand


def test_mlpblock_preserves_shape():
    blk = MLPBlock(8, 2, key=key())
    assert blk(rand((8,), 1), rand((8,), 2)).shape == (8,)


def test_mlpblock_takes_one_example_and_vmaps():
    blk = MLPBlock(8, 2, key=key())
    assert blk(rand((8,), 1), rand((8,), 2)).shape == (8,)
    assert jax.vmap(blk)(rand((5, 8), 1), rand((5, 8), 2)).shape == (5, 8)


def test_mlpblock_rowwise_independent():
    blk = MLPBlock(8, 2, key=key())
    x, c = rand((4, 8), 1), rand((4, 8), 2)
    close(jax.vmap(blk)(x, c), jnp.stack([blk(x[i], c[i]) for i in range(4)]))


def test_mlpblock_expand_one():
    blk = MLPBlock(8, 1, key=key())
    assert blk(rand((8,), 1), rand((8,), 2)).shape == (8,)


def test_mlpblock_dim_one():
    blk = MLPBlock(1, 2, key=key())
    assert blk(rand((1,), 1), rand((1,), 2)).shape == (1,)


def test_mlpblock_is_residual_identity_at_init():
    blk = MLPBlock(8, 2, key=key())
    x = rand((8,), 1)
    close(blk(x, rand((8,), 2)), x)


def test_mlpblock_finite_on_large_input():
    blk = MLPBlock(8, 2, key=key())
    out = blk(1e4 * rand((8,), 1), rand((8,), 2))
    assert bool(jnp.all(jnp.isfinite(out)))


def test_mlpblock_identity_at_init_is_bit_exact():
    blk = MLPBlock(8, 2, key=key())
    x = rand((8,), 1)
    close(blk(x, rand((8,), 2)), x, atol=0.0)


def test_mlpblock_depends_on_c_with_perturbed_params():
    blk = perturbed(MLPBlock(8, 2, key=key()))
    x = rand((8,), 1)
    differs(blk(x, rand((8,), 2)), blk(x, rand((8,), 6)))


def test_mlpblock_residual_survives_perturbation():
    blk = perturbed(MLPBlock(8, 2, key=key()))
    x = rand((8,), 1)
    out = blk(x, rand((8,), 2))
    assert out.shape == x.shape
    assert bool(jnp.all(jnp.isfinite(out)))


def test_mlpblock_expand_widens_the_hidden_layer():
    narrow = MLPBlock(8, 1, key=key())
    wide = MLPBlock(8, 4, key=key())
    assert narrow.mlp.linear2.weight.shape == (8, 8)
    assert wide.mlp.linear2.weight.shape == (8, 32)
