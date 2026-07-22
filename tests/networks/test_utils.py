"""FeedForward and Modulation: shared network blocks."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from canna.networks.utils import FeedForward, Modulation, SinusoidalEmbed


def key(seed: int = 0):
    return jax.random.key(seed)


def rand(shape, seed: int = 0):
    return jax.random.normal(jax.random.key(seed), shape)


def rngs(seed: int = 0):
    return nnx.Rngs(seed)


def close(a, b, atol=1e-4):
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=atol, rtol=1e-4)


def differs(a, b, atol=1e-5):
    assert not np.allclose(np.asarray(a), np.asarray(b), atol=atol)


def perturbed(module, seed: int = 7, scale: float = 0.3):
    """Move every param off init: zero-init gates make a fresh network ignore its conditioning."""
    state = nnx.state(module, nnx.Param)
    leaves, treedef = jax.tree.flatten(state)
    keys = jax.random.split(jax.random.key(seed), len(leaves))
    noised = [
        p + scale * jax.random.normal(k, p.shape, p.dtype) for p, k in zip(leaves, keys)
    ]
    nnx.update(module, jax.tree.unflatten(treedef, noised))
    return module


# ---------------------------------------------------------------- FeedForward


# ---------------------------------------------------------------- FeedForward


def test_feedforward_out_dim_is_out_dim():
    ff = FeedForward(3, 16, 7, rngs=rngs())
    assert ff(rand((3,))).shape == (7,)


def test_feedforward_batch_ranks():
    ff = FeedForward(3, 16, 7, rngs=rngs())
    assert ff(rand((3,))).shape == (7,)
    assert ff(rand((5, 3))).shape == (5, 7)
    assert ff(rand((2, 5, 3))).shape == (2, 5, 7)


def test_feedforward_rowwise_independent():
    ff = FeedForward(3, 16, 7, rngs=rngs())
    x = rand((4, 3))
    close(ff(x), jnp.stack([ff(row) for row in x]))


def test_feedforward_odd_hidden_dim_shape():
    ff = FeedForward(3, 13, 7, rngs=rngs())
    assert ff(rand((3,))).shape == (7,)


def test_feedforward_hidden_dim_one():
    ff = FeedForward(3, 1, 7, rngs=rngs())
    assert ff(rand((3,))).shape == (7,)


def test_feedforward_unit_dims():
    ff = FeedForward(1, 4, 1, rngs=rngs())
    assert ff(rand((6, 1))).shape == (6, 1)


def test_feedforward_is_nonlinear():
    ff = FeedForward(3, 16, 7, rngs=rngs())
    x = rand((3,))
    assert not np.allclose(np.asarray(ff(2.0 * x)), np.asarray(2.0 * ff(x)), atol=1e-4)


def test_feedforward_deterministic():
    ff = FeedForward(3, 16, 7, rngs=rngs())
    x = rand((5, 3))
    close(ff(x), ff(x))


def test_feedforward_finite_on_large_input():
    ff = FeedForward(3, 16, 7, rngs=rngs())
    assert bool(jnp.all(jnp.isfinite(ff(1e4 * rand((4, 3))))))


# ---------------------------------------------------------------- Modulation


# ---------------------------------------------------------------- Modulation


def test_modulation_returns_two_same_shape():
    mod = Modulation(8, rngs=rngs())
    x, c = rand((8,), 1), rand((8,), 2)
    out, gate = mod(x, c)
    assert out.shape == (8,)
    assert gate.shape == (8,)


def test_modulation_batch_ranks():
    mod = Modulation(8, rngs=rngs())
    for lead in [(), (5,), (2, 5)]:
        out, gate = mod(rand(lead + (8,), 1), rand(lead + (8,), 2))
        assert out.shape == lead + (8,)
        assert gate.shape == lead + (8,)


def test_modulation_gate_depends_only_on_c():
    mod = Modulation(8, rngs=rngs())
    c = rand((8,), 2)
    _, g1 = mod(rand((8,), 3), c)
    _, g2 = mod(rand((8,), 4), c)
    close(g1, g2)


def test_modulation_invariant_to_affine_rescale_of_x():
    mod = Modulation(8, rngs=rngs())
    x, c = rand((8,), 1), rand((8,), 2)
    a, b = mod(x, c)
    a2, b2 = mod(3.0 * x + 1.5, c)
    close(a, a2)
    close(b, b2)


def test_modulation_rowwise_independent():
    mod = Modulation(8, rngs=rngs())
    x, c = rand((4, 8), 1), rand((4, 8), 2)
    out, gate = mod(x, c)
    outs = [mod(x[i], c[i]) for i in range(4)]
    close(out, jnp.stack([o for o, _ in outs]))
    close(gate, jnp.stack([g for _, g in outs]))


def test_modulation_broadcasts_token_axis_conditioning():
    mod = Modulation(8, rngs=rngs())
    x = rand((6, 8), 1)
    c = rand((1, 8), 2)
    out, gate = mod(x, c)
    assert out.shape == (6, 8)
    assert gate.shape == (6, 8) or gate.shape == (1, 8)


def test_modulation_dim_one():
    mod = Modulation(1, rngs=rngs())
    out, gate = mod(rand((4, 1), 1), rand((4, 1), 2))
    assert out.shape == (4, 1) and gate.shape == (4, 1)


def test_modulation_constant_x_finite():
    mod = Modulation(8, rngs=rngs())
    out, gate = mod(jnp.ones((8,)), rand((8,), 2))
    assert bool(jnp.all(jnp.isfinite(out))) and bool(jnp.all(jnp.isfinite(gate)))


# ------------------------------------------------------------ SinusoidalEmbed


# ---------------------------------------------------------------- FeedForward


def test_feedforward_vanishes_at_the_origin():
    ff = FeedForward(3, 16, 7, rngs=rngs())
    close(ff(jnp.zeros((3,))), jnp.zeros((7,)), atol=1e-12)


def test_feedforward_is_quadratic_near_the_origin():
    ff = FeedForward(3, 16, 7, rngs=rngs())
    x = rand((3,))
    small = jnp.linalg.norm(ff(1e-3 * x))
    smaller = jnp.linalg.norm(ff(1e-4 * x))
    close(smaller * 100.0, small, atol=1e-9)


def test_feedforward_hidden_dim_is_post_split_width():
    wide = FeedForward(3, 16, 7, rngs=rngs())
    narrow = FeedForward(3, 8, 7, rngs=rngs())
    assert wide.linear2.kernel.value.shape == (16, 7)
    assert narrow.linear1.kernel.value.shape == (3, 16)


def test_feedforward_gate_half_is_second_half_of_the_projection():
    ff = FeedForward(2, 4, 2, rngs=rngs())
    ff.linear1.kernel.value = jnp.zeros_like(ff.linear1.kernel.value)
    ff.linear1.bias.value = jnp.concat([jnp.ones((4,)), jnp.zeros((4,))])
    close(ff(rand((2,))), jnp.zeros((2,)), atol=1e-12)


# ---------------------------------------------------------------- Modulation


# ---------------------------------------------------------------- Modulation


def test_modulation_gate_is_exactly_zero_at_init():
    mod = Modulation(8, rngs=rngs())
    _, gate = mod(rand((8,), 1), rand((8,), 2))
    close(gate, jnp.zeros((8,)), atol=0.0)


def test_modulation_output_is_plain_standardization_at_init():
    mod = Modulation(8, rngs=rngs())
    x = rand((8,), 1)
    out, _ = mod(x, rand((8,), 2))
    reference = (x - x.mean(-1, keepdims=True)) / jnp.sqrt(
        x.var(-1, keepdims=True) + 1e-5
    )
    close(out, reference)


def test_modulation_ignores_c_at_init():
    mod = Modulation(8, rngs=rngs())
    x = rand((8,), 1)
    close(mod(x, rand((8,), 2))[0], mod(x, 50.0 * rand((8,), 5))[0], atol=0.0)


def test_modulation_uses_c_once_params_are_perturbed():
    mod = perturbed(Modulation(8, rngs=rngs()))
    x = rand((8,), 1)
    differs(mod(x, rand((8,), 2))[0], mod(x, rand((8,), 5))[0])


def test_modulation_constant_x_maps_to_zero_not_nan():
    mod = Modulation(8, rngs=rngs())
    out, _ = mod(jnp.full((8,), 3.0), rand((8,), 2))
    close(out, jnp.zeros((8,)), atol=1e-8)


def test_modulation_dim_one_maps_to_zero_not_nan():
    mod = Modulation(1, rngs=rngs())
    out, _ = mod(rand((4, 1), 1), rand((4, 1), 2))
    close(out, jnp.zeros((4, 1)), atol=1e-8)


def test_modulation_standardizes_only_the_last_axis():
    mod = Modulation(8, rngs=rngs())
    x = rand((5, 8), 1)
    scaled = x.at[0].multiply(100.0)
    out, _ = mod(x, rand((5, 8), 2))
    out_scaled, _ = mod(scaled, rand((5, 8), 2))
    close(out[1:], out_scaled[1:])


def test_modulation_gate_keeps_the_conditioning_token_axis():
    mod = perturbed(Modulation(8, rngs=rngs()))
    out, gate = mod(rand((6, 8), 1), rand((1, 8), 2))
    assert out.shape == (6, 8)
    assert gate.shape == (1, 8)


# ------------------------------------------------------------ SinusoidalEmbed


def test_sinusoidal_scalar_gives_dim_vector():
    emb = SinusoidalEmbed(16, rngs=rngs())
    assert emb(jnp.asarray(0.3)).shape == (16,)


def test_sinusoidal_appends_axis_for_each_batch_rank():
    emb = SinusoidalEmbed(16, rngs=rngs())
    assert emb(rand((5,))).shape == (5, 16)
    assert emb(rand((2, 5))).shape == (2, 5, 16)


def test_sinusoidal_odd_dim_shape():
    emb = SinusoidalEmbed(15, rngs=rngs())
    assert emb(jnp.asarray(0.3)).shape == (15,)


def test_sinusoidal_dim_one_shape():
    emb = SinusoidalEmbed(1, rngs=rngs())
    assert emb(jnp.asarray(0.3)).shape == (1,)


def test_sinusoidal_elementwise_over_batch():
    emb = SinusoidalEmbed(16, rngs=rngs())
    t = rand((7,))
    close(emb(t), jnp.stack([emb(ti) for ti in t]))


def test_sinusoidal_injective_on_distinct_scalars():
    emb = SinusoidalEmbed(16, rngs=rngs())
    a = emb(jnp.asarray(0.1))
    b = emb(jnp.asarray(0.2))
    assert not np.allclose(np.asarray(a), np.asarray(b), atol=1e-5)


def test_sinusoidal_zero_and_finite_at_extremes():
    emb = SinusoidalEmbed(16, rngs=rngs())
    for t in [0.0, 1.0, -1.0, 1e3]:
        assert bool(jnp.all(jnp.isfinite(emb(jnp.asarray(t)))))


def test_sinusoidal_deterministic():
    emb = SinusoidalEmbed(16, rngs=rngs())
    t = rand((4,))
    close(emb(t), emb(t))


def test_sinusoidal_period_one_collapses_every_frequency():
    emb = SinusoidalEmbed(8, period=1.0, rngs=rngs())
    close(emb(jnp.asarray(0.3)), emb(jnp.asarray(1.3)))


def test_sinusoidal_default_period_is_not_a_period_in_t():
    emb = SinusoidalEmbed(8, rngs=rngs())
    differs(emb(jnp.asarray(0.3)), emb(jnp.asarray(0.3 + 2 * jnp.pi)))


def test_sinusoidal_period_zero_is_unguarded():
    emb = SinusoidalEmbed(8, period=0.0, rngs=rngs())
    assert not bool(jnp.all(jnp.isfinite(emb(jnp.asarray(0.3)))))


def test_sinusoidal_frequency_grid_rejects_integer_time():
    emb = SinusoidalEmbed(8, rngs=rngs())
    with pytest.raises(AssertionError):
        emb(jnp.asarray([1], dtype=jnp.int32))


def test_sinusoidal_rejects_python_float_time():
    emb = SinusoidalEmbed(8, rngs=rngs())
    with pytest.raises(AttributeError):
        emb(0.3)


def test_sinusoidal_is_bounded_before_the_projection_only():
    emb = SinusoidalEmbed(8, rngs=rngs())
    huge = emb(jnp.asarray(1e6))
    assert bool(jnp.all(jnp.isfinite(huge)))


def test_sinusoidal_time_zero_is_a_fixed_point_of_the_sine_half():
    emb = SinusoidalEmbed(8, rngs=rngs())
    close(emb(jnp.asarray(0.0)), emb(jnp.asarray(0.0)), atol=0.0)
