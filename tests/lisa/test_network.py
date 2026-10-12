"""LisaFlow: shapes, permutation invariance over binaries, the conditioning, and the PLE."""

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


def test_velocity_shape():
    assert flow()(*inputs()).shape == (S, X)


def test_takes_one_example_and_vmaps():
    dx = jax.vmap(flow())(
        rand((3, S, X), 1), rand((3,), 3), rand((3, BINS, C), 2), rand((3,), 4) * 1e-4 + F
    )
    assert dx.shape == (3, S, X)


def test_batch_rows_are_independent():
    net = perturbed(flow())
    x, t = rand((3, S, X), 1), rand((3,), 3)
    y, f = rand((3, BINS, C), 2), rand((3,), 4) * 1e-4 + F
    dx = jax.vmap(net)(x, t, y, f)
    # float32 inside, and the batch changes the summation order of the large perturbed sums
    close(dx, jnp.stack([net(x[i], t[i], y[i], f[i]) for i in range(3)]), atol=2e-3)


@pytest.mark.parametrize("sources", [1, 2, 7])
def test_source_count_is_free(sources):
    x, t, y, f = inputs()
    assert flow()(rand((sources, X), 1), t, y, f).shape == (sources, X)


@pytest.mark.parametrize("t", [0.0, 1.0])
def test_finite_at_the_time_endpoints(t):
    x, _, y, f = inputs()
    assert bool(jnp.all(jnp.isfinite(flow()(x, jnp.asarray(t), y, f))))


def test_ignores_conditioning_at_init():
    # zero-init gates: a fresh network's velocity is a function of the sources alone
    net, x = flow(), rand((S, X), 1)
    a = net(x, jnp.asarray(0.1), rand((BINS, C), 2), jnp.asarray(F))
    b = net(x, jnp.asarray(0.9), 10.0 * rand((BINS, C), 9), jnp.asarray(9.0e-3))
    close(a, b, atol=0.0)


def test_velocity_depends_on_t_once_perturbed():
    net, (x, _, y, f) = perturbed(flow()), inputs()
    differs(net(x, jnp.asarray(0.1), y, f), net(x, jnp.asarray(0.9), y, f))


def test_velocity_depends_on_y_once_perturbed():
    net, (x, t, _, f) = perturbed(flow()), inputs()
    differs(net(x, t, rand((BINS, C), 2), f), net(x, t, rand((BINS, C), 12), f))


def test_velocity_depends_on_where_the_signal_sits_once_perturbed():
    """The y tokens carry a learned position: moving a feature along the window matters."""
    net, (x, t, _, f) = perturbed(flow()), inputs()
    y = jnp.zeros((BINS, C)).at[2].set(5.0)
    differs(net(x, t, y, f), net(x, t, jnp.roll(y, 6, axis=0), f))


def test_velocity_depends_on_f_once_perturbed():
    """f is the window's centre frequency: the flow must know which band it is looking at"""
    net, (x, t, y, _) = perturbed(flow()), inputs()
    differs(net(x, t, y, jnp.asarray(1.0e-3)), net(x, t, y, jnp.asarray(9.0e-3)))


def test_permutation_equivariant_over_sources_once_perturbed():
    net = perturbed(flow())
    x, t, y, f = inputs()
    perm = jnp.asarray([3, 1, 0, 2])
    close(net(x[perm], t, y, f), net(x, t, y, f)[perm])


def test_x_stream_has_no_positional_embedding():
    net = perturbed(flow())
    _, t, y, f = inputs()
    token = rand((1, X), 1)
    dx = net(jnp.concat([token, token + 1e-9], axis=0), t, y, f)
    close(dx[0], dx[1], atol=1e-5)


def test_velocity_is_differentiable_in_x():
    net, (x, t, y, f) = perturbed(flow()), inputs()
    g = jax.jacobian(lambda z: net(z, t, y, f))(x)
    assert g.shape == (S, X, S, X) and bool(jnp.all(jnp.isfinite(g)))


def test_zero_blocks():
    assert flow(blocks=0)(*inputs()).shape == (S, X)


def test_bf16_backbone_keeps_float32_params_and_output():
    net = flow(dtype=jnp.bfloat16)
    dx = perturbed(net)(*inputs())
    assert dx.dtype == jnp.float32
    assert all(p.dtype == jnp.float32 for p in jax.tree.leaves(eqx.filter(net, eqx.is_array)))


def test_no_biases_inside_the_blocks():
    biases = [b for b in jax.tree.leaves(flow().backbone, is_leaf=lambda l: l is None)]
    assert all(b is None or b.ndim >= 3 for b in biases)
    linear = flow().backbone.blocks.mlp_x.linear1
    assert linear.bias is None


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
