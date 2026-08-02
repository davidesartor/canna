"""Fold2d/Unfold2d and the Patchify/UnPatchify stack built on them."""

import jax
import jax.numpy as jnp
import pytest

from canna.networks import (
    Fold2d,
    Unfold2d,
    Patchify,
    UnPatchify,
)

from ._helpers import close, key, rand

# ---------------------------------------------------------------- Fold2d


def test_fold2d_shape_halves_space_quadruples_channels():
    f = Fold2d()
    assert f(rand((8, 6, 3))).shape == (4, 3, 12)


def test_fold2d_takes_one_image_and_vmaps():
    f = Fold2d()
    assert f(rand((8, 6, 3))).shape == (4, 3, 12)
    assert jax.vmap(f)(rand((5, 8, 6, 3))).shape == (5, 4, 3, 12)


def test_fold2d_channel_layout_is_patch_row_major():
    f = Fold2d()
    z = jnp.arange(2 * 2 * 3, dtype=jnp.float64).reshape(2, 2, 3)
    close(f(z)[0, 0], z.reshape(-1))


def test_fold2d_equivariant_to_block_permutation():
    f = Fold2d()
    z = rand((4, 4, 2))
    swapped = jnp.concatenate([z[2:4], z[0:2]], axis=0)
    close(f(swapped), f(z)[::-1])


def test_fold2d_rejects_odd_spatial_dims():
    f = Fold2d()
    with pytest.raises(Exception):
        f(rand((5, 4, 2)))


# ---------------------------------------------------------------- Unfold2d


def test_unfold2d_shape_doubles_space_quarters_channels():
    u = Unfold2d()
    assert u(rand((4, 3, 12))).shape == (8, 6, 3)


def test_unfold2d_takes_one_image_and_vmaps():
    u = Unfold2d()
    assert u(rand((4, 3, 12))).shape == (8, 6, 3)
    assert jax.vmap(u)(rand((5, 4, 3, 12))).shape == (5, 8, 6, 3)


def test_unfold2d_channel_layout_is_patch_row_major():
    u = Unfold2d()
    z = jnp.arange(12, dtype=jnp.float64).reshape(1, 1, 12)
    close(u(z), z.reshape(2, 2, 3))


def test_unfold_of_fold_is_identity():
    f, u = Fold2d(), Unfold2d()
    z = rand((8, 6, 3))
    close(u(f(z)), z)


def test_fold_unfold_roundtrip_survives_vmap():
    f, u = Fold2d(), Unfold2d()
    z = rand((2, 3, 4, 4, 2))
    roundtrip = jax.vmap(jax.vmap(lambda w: u(f(w))))
    close(roundtrip(z), z, atol=0.0)


def test_unfold2d_rejects_channels_not_divisible_by_four():
    u = Unfold2d()
    with pytest.raises(Exception):
        u(rand((4, 3, 6)))


# ---------------------------------------------------------------- Patchify


@pytest.mark.parametrize("stages", [1, 2, 3])
def test_patchify_downsamples_by_exactly_two_per_stage(stages):
    p = Patchify(3, 64, stages=stages, key=key())
    assert p(rand((16, 16, 3))).shape == (16 >> stages, 16 >> stages, 64)


def test_patchify_default_stages_is_four():
    p = Patchify(3, 64, key=key())
    assert p(rand((16, 16, 3))).shape == (1, 1, 64)


def test_patchify_takes_one_image_and_vmaps():
    p = Patchify(3, 32, stages=2, key=key())
    assert p(rand((16, 8, 3))).shape == (4, 2, 32)
    assert jax.vmap(p)(rand((5, 16, 8, 3))).shape == (5, 4, 2, 32)


def test_patchify_batch_independent():
    p = Patchify(3, 32, stages=2, key=key())
    y = rand((4, 16, 8, 3))
    close(jax.vmap(p)(y), jnp.stack([p(y[i]) for i in range(4)]))


def test_patchify_holds_no_positional_information():
    p = Patchify(3, 64, stages=2, key=key())
    y = rand((16, 16, 3))
    close(p(jnp.roll(y, 8, axis=-2)), jnp.roll(p(y), 2, axis=-2))


def test_patchify_is_token_local():
    p = Patchify(3, 64, stages=1, key=key())
    y = rand((4, 4, 3))
    edited = y.at[0, 0].set(y[0, 0] + 5.0)
    close(p(y)[1:], p(edited)[1:])


def test_patchify_rejects_indivisible_resolution():
    p = Patchify(3, 32, stages=2, key=key())
    with pytest.raises(Exception):
        p(rand((14, 8, 3)))


def test_patchify_asserts_dim_divisible_by_narrowest_stage():
    with pytest.raises(AssertionError, match="hidden_dim must be divisible by 64"):
        Patchify(3, 32, stages=4, key=key())


def test_patchify_zero_stages_is_rejected_by_the_guard():
    with pytest.raises(AssertionError):
        Patchify(3, 32, stages=0, key=key())


# ---------------------------------------------------------------- UnPatchify


@pytest.mark.parametrize("stages", [1, 2, 3])
def test_unpatchify_upsamples_by_exactly_two_per_stage(stages):
    u = UnPatchify(3, 64, stages=stages, key=key())
    assert u(rand((2, 2, 64))).shape == (2 << stages, 2 << stages, 3)


@pytest.mark.parametrize("stages", [1, 2, 3])
def test_unpatchify_inverts_patchify_shape_for_every_stage_count(stages):
    p = Patchify(3, 64, stages=stages, key=key())
    u = UnPatchify(3, 64, stages=stages, key=key())
    y = rand((16, 16, 3))
    assert u(p(y)).shape == y.shape


def test_unpatchify_default_stages_is_four():
    u = UnPatchify(3, 64, key=key())
    assert u(rand((1, 1, 64))).shape == (16, 16, 3)


def test_unpatchify_takes_one_image_and_vmaps():
    u = UnPatchify(3, 32, stages=2, key=key())
    assert u(rand((4, 2, 32))).shape == (16, 8, 3)
    assert jax.vmap(u)(rand((5, 4, 2, 32))).shape == (5, 16, 8, 3)


def test_unpatchify_batch_independent():
    u = UnPatchify(3, 32, stages=2, key=key())
    z = rand((4, 4, 2, 32))
    close(jax.vmap(u)(z), jnp.stack([u(z[i]) for i in range(4)]))


def test_unpatchify_is_token_local():
    u = UnPatchify(3, 64, stages=1, key=key())
    z = rand((2, 2, 64))
    edited = z.at[0, 0].set(z[0, 0] + 5.0)
    close(u(z)[2:], u(edited)[2:])


def test_unpatchify_asserts_dim_divisible_by_narrowest_stage():
    with pytest.raises(AssertionError, match="hidden_dim must be divisible by 64"):
        UnPatchify(3, 32, stages=4, key=key())


def test_unpatchify_zero_stages_is_rejected_by_the_guard():
    with pytest.raises(AssertionError):
        UnPatchify(3, 32, stages=0, key=key())
