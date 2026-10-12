"""LisaFlow: shapes, labelled source slots, the conditioning, the two heads, and the PLE."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from canna.lisa import LisaFlow
from canna.lisa.network import arcsinh_edges
from canna.networks import PiecewiseLinear

S, X, BINS, C, HIDDEN, HEADS = 4, 11, 16, 6, 16, 2
F, F_RANGE = 3.0e-3, (1.0e-4, 1.2e-2)


def rand(shape, seed: int = 0):
    return jax.random.normal(jax.random.key(seed), shape)


def close(a, b, atol=1e-4):
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=atol, rtol=1e-4)


def differs(a, b, atol=1e-5):
    assert not np.allclose(np.asarray(a), np.asarray(b), atol=atol)


def flow(
    sources=S, bins=BINS, hidden=HIDDEN, heads=HEADS, blocks=2, patch=2, seed=0, dtype=jnp.float32
):
    return LisaFlow(
        (sources, X),
        (bins, C),
        hidden,
        heads,
        blocks,
        F_RANGE,
        patch,
        dtype=dtype,
        key=jax.random.key(seed),
    )


def perturbed(module, seed: int = 7, scale: float = 0.3):
    """Move every param off init: zero-init gates make a fresh network ignore its conditioning."""
    params, static = eqx.partition(module, eqx.is_inexact_array)
    leaves, treedef = jax.tree.flatten(params)
    keys = jax.random.split(jax.random.key(seed), len(leaves))
    noised = [
        p + scale * jax.random.normal(k, p.shape, p.dtype) for p, k in zip(leaves, keys)
    ]
    return eqx.combine(jax.tree.unflatten(treedef, noised), static)


def inputs(seed=1):
    return rand((S, X), seed), jnp.asarray(0.3), rand((BINS, C), seed + 1), jnp.asarray(F)


def test_output_shapes():
    dx, x1 = flow()(*inputs())
    assert dx.shape == (S, X) and x1.shape == (S, X)


def test_takes_one_example_and_vmaps():
    dx, x1 = jax.vmap(flow())(
        rand((3, S, X), 1), rand((3,), 3), rand((3, BINS, C), 2), rand((3,), 4) * 1e-4 + F
    )
    assert dx.shape == (3, S, X) and x1.shape == (3, S, X)


def test_batch_rows_are_independent():
    net = perturbed(flow())
    x, t = rand((3, S, X), 1), rand((3,), 3)
    y, f = rand((3, BINS, C), 2), rand((3,), 4) * 1e-4 + F
    dx = jax.vmap(net)(x, t, y, f)[0]
    # float32 inside, and the batch changes the summation order of the large perturbed sums
    close(dx, jnp.stack([net(x[i], t[i], y[i], f[i])[0] for i in range(3)]), atol=2e-3)


def test_the_source_count_is_the_slot_count():
    x, t, y, f = inputs()
    with pytest.raises(AssertionError, match="source slots"):
        flow()(rand((S - 1, X), 1), t, y, f)


@pytest.mark.parametrize("t", [0.0, 1.0])
def test_finite_at_the_time_endpoints(t):
    x, _, y, f = inputs()
    assert all(bool(jnp.all(jnp.isfinite(o))) for o in flow()(x, jnp.asarray(t), y, f))


def test_velocity_and_reconstruction_are_distinct_heads():
    dx, x1 = perturbed(flow())(*inputs())
    differs(dx, x1)


def test_ignores_conditioning_at_init():
    # zero-init gates: a fresh network's outputs are a function of the sources alone
    net, x = flow(), rand((S, X), 1)
    a = net(x, jnp.asarray(0.1), rand((BINS, C), 2), jnp.asarray(F))
    b = net(x, jnp.asarray(0.9), 10.0 * rand((BINS, C), 9), jnp.asarray(9.0e-3))
    close(a, b, atol=0.0)


def test_velocity_depends_on_t_once_perturbed():
    net, (x, _, y, f) = perturbed(flow()), inputs()
    differs(net(x, jnp.asarray(0.1), y, f)[0], net(x, jnp.asarray(0.9), y, f)[0])


def test_velocity_depends_on_y_once_perturbed():
    net, (x, t, _, f) = perturbed(flow()), inputs()
    differs(net(x, t, rand((BINS, C), 2), f)[0], net(x, t, rand((BINS, C), 12), f)[0])


def test_velocity_depends_on_where_the_signal_sits_once_perturbed():
    """The y tokens carry a learned position: moving a feature along the window matters."""
    net, (x, t, _, f) = perturbed(flow()), inputs()
    y = jnp.zeros((BINS, C)).at[2].set(5.0)
    differs(net(x, t, y, f)[0], net(x, t, jnp.roll(y, 6, axis=0), f)[0])


def test_velocity_depends_on_f_once_perturbed():
    """f is the window's centre frequency: the flow must know which band it is looking at"""
    net, (x, t, y, _) = perturbed(flow()), inputs()
    differs(net(x, t, y, jnp.asarray(1.0e-3))[0], net(x, t, y, jnp.asarray(9.0e-3))[0])


def test_source_slots_are_labelled_once_perturbed():
    """Sources come loudest first, so two identical sources in different slots are told apart."""
    net = perturbed(flow())
    _, t, y, f = inputs()
    token = rand((1, X), 1)
    dx = net(jnp.concat([token] * S, axis=0), t, y, f)[0]
    differs(dx[0], dx[1])


def test_velocity_is_differentiable_in_x():
    net, (x, t, y, f) = perturbed(flow()), inputs()
    g = jax.jacobian(lambda z: net(z, t, y, f)[0])(x)
    assert g.shape == (S, X, S, X) and bool(jnp.all(jnp.isfinite(g)))


def test_zero_blocks():
    assert flow(blocks=0)(*inputs())[0].shape == (S, X)


def test_bf16_backbone_keeps_float32_params_and_output():
    net = flow(dtype=jnp.bfloat16)
    dx, x1 = perturbed(net)(*inputs())
    assert dx.dtype == jnp.float32 and x1.dtype == jnp.float32
    assert all(p.dtype == jnp.float32 for p in jax.tree.leaves(eqx.filter(net, eqx.is_array)))


def test_no_biases_inside_the_blocks():
    assert flow().backbone.blocks.mlp_x.linear1.bias is None
    assert flow().backbone.blocks.mod_x_attn.linear.bias is None


def test_rejects_a_window_the_patch_does_not_divide():
    with pytest.raises(AssertionError, match="not divisible by 2"):
        flow(bins=15)


def test_ple_ramps_through_its_bins_and_extrapolates_at_the_ends():
    ple = PiecewiseLinear(((0.0, 1.0, 2.0, 4.0),), 3, key=jax.random.key(0))
    close(ple.encode(jnp.asarray([1.5])), [1.0, 0.5, 0.0], atol=1e-7)
    close(ple.encode(jnp.asarray([3.0])), [1.0, 1.0, 0.5], atol=1e-7)
    # past the outer edges the first and last components keep ramping
    close(ple.encode(jnp.asarray([-1.0])), [-1.0, 0.0, 0.0], atol=1e-7)
    close(ple.encode(jnp.asarray([8.0])), [1.0, 1.0, 3.0], atol=1e-7)


def test_ple_is_continuous_and_takes_ragged_edges():
    ple = PiecewiseLinear(((0.0, 1.0, 2.0), (-1.0, 0.0, 0.5, 1.0)), 4, key=jax.random.key(0))
    a = ple.encode(jnp.asarray([1.0 - 1e-6, 0.5]))
    b = ple.encode(jnp.asarray([1.0 + 1e-6, 0.5]))
    assert a.shape == (5,)
    close(a, b, atol=1e-5)


def test_data_edges_are_symmetric_and_fine_near_the_noise():
    edges = np.asarray(arcsinh_edges(1000.0, 64))
    close(edges, -edges[::-1], atol=1e-9)
    assert np.isclose(edges[-1], 1000.0)
    assert np.diff(edges).min() < 0.25 and np.diff(edges).max() < 250
