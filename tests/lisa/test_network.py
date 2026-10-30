"""LisaFlow: shapes, permutation equivariance, the conditioning, the frame head, the cached data stream and the attention sinks."""

from functools import partial
from types import SimpleNamespace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from canna.lisa.network import LisaFlow

S, X, BINS, C, HIDDEN, HEADS = 4, 8, 64, 6, 16, 2
F, F_RANGE = 3.0e-3, (1.0e-4, 1.2e-2)


def rand(shape, seed: int = 0):
    return jax.random.normal(jax.random.key(seed), shape)


def close(a, b, atol=1e-4):
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=atol, rtol=1e-4)


def differs(a, b, atol=1e-5):
    assert not np.allclose(np.asarray(a), np.asarray(b), atol=atol)


def flow(
    sources=S,
    bins=BINS,
    hidden=HIDDEN,
    heads=HEADS,
    blocks=2,
    seed=0,
    dtype=jnp.float32,
    sinks=4,
):
    # LisaFlow reads only the window's layout off the problem
    problem = SimpleNamespace(
        n_sources=sources,
        window_bins=bins,
        t_obs=63115200.0,
        response_points=16,
        f0_range=F_RANGE,
    )
    return LisaFlow(
        problem,
        hidden,
        heads,
        blocks,
        sinks=sinks,
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


def run(net, z, t, y, f):
    """The flow's (m, s) at z, in the frame its own encoding of (y, f) reads off delta."""
    delta, cache = net.encode(y, f)
    return net(z, t, (delta[:, :X], delta[:, X:]), f, cache)


def inputs(seed=1):
    return (
        rand((S, X), seed),
        jnp.asarray(0.3),
        rand((BINS, C), seed + 1),
        jnp.asarray(F),
    )


def test_output_shapes():
    assert run(flow(), *inputs()).shape == (S, 2 * X)
    assert flow().encode(*inputs()[2:])[0].shape == (S, 2 * X)


def test_takes_one_example_and_vmaps():
    out = jax.vmap(partial(run, flow()))(
        rand((3, S, X), 1),
        rand((3,), 3),
        rand((3, BINS, C), 2),
        rand((3,), 4) * 1e-4 + F,
    )
    assert out.shape == (3, S, 2 * X)


def test_batch_rows_are_independent():
    net = perturbed(flow())
    x, t = rand((3, S, X), 1), rand((3,), 3)
    y, f = rand((3, BINS, C), 2), rand((3,), 4) * 1e-4 + F
    out = jax.vmap(partial(run, net))(x, t, y, f)
    # float32 inside, and the batch changes the summation order of the large perturbed sums
    close(out, jnp.stack([run(net, x[i], t[i], y[i], f[i]) for i in range(3)]), atol=2e-3)


@pytest.mark.parametrize("t", [0.0, 1.0])
def test_finite_at_the_time_endpoints(t):
    x, _, y, f = inputs()
    assert bool(jnp.all(jnp.isfinite(run(flow(), x, jnp.asarray(t), y, f))))


def test_one_encoding_serves_every_draw():
    net, (_, t, y, f) = perturbed(flow()), inputs()
    (delta, cache), zs = net.encode(y, f), rand((3, S, X), 5)
    frame = (delta[:, :X], delta[:, X:])
    shared = jax.vmap(lambda z: net(z, t, frame, f, cache))(zs)
    close(shared, jnp.stack([run(net, z, t, y, f) for z in zs]), atol=2e-3)


def test_sinks_run_per_draw_and_change_the_flow_once_perturbed():
    plain, sunk = perturbed(flow(sinks=0)), perturbed(flow(sinks=3))
    x, t, y, f = inputs()
    # the sinks are not part of the data stream's cache
    assert sunk.encode(y, f)[1][0].shape == plain.encode(y, f)[1][0].shape
    out = run(sunk, x, t, y, f)
    assert out.shape == (S, 2 * X) and bool(jnp.all(jnp.isfinite(out)))
    differs(out, run(plain, x, t, y, f))


def test_ignores_t_and_y_at_init():
    # zero-init gates: a fresh network's outputs are a function of the sources (and, through f0's bin, f) alone
    net, x = flow(), rand((S, X), 1)
    a = run(net, x, jnp.asarray(0.1), rand((BINS, C), 2), jnp.asarray(F))
    b = run(net, x, jnp.asarray(0.9), 10.0 * rand((BINS, C), 9), jnp.asarray(F))
    close(a, b, atol=0.0)


def test_the_frame_starts_at_the_prior():
    """No bias on the frame head and a zero-init modulation: delta is 0 at init, whatever the data."""
    net = flow()
    assert net.frame_unembed.bias is None
    delta, _ = net.encode(10.0 * rand((BINS, C), 2), jnp.asarray(F))
    close(delta, jnp.zeros((S, 2 * X)), atol=0.0)


def test_the_frame_depends_on_y_once_perturbed():
    net, (_, _, y, f) = perturbed(flow()), inputs()
    differs(net.encode(y, f)[0], net.encode(rand((BINS, C), 12), f)[0])


def test_the_late_time_embedding_resolves_the_last_decades_once_perturbed():
    net, (x, _, y, f) = perturbed(flow()), inputs()
    late = lambda t: run(net, x, jnp.asarray(1 - t), y, f)
    differs(late(1e-4), late(1e-5))


def test_velocity_depends_on_t_once_perturbed():
    net, (x, _, y, f) = perturbed(flow()), inputs()
    differs(run(net, x, jnp.asarray(0.1), y, f), run(net, x, jnp.asarray(0.9), y, f))


def test_velocity_depends_on_y_once_perturbed():
    net, (x, t, _, f) = perturbed(flow()), inputs()
    differs(run(net, x, t, rand((BINS, C), 2), f), run(net, x, t, rand((BINS, C), 12), f))


def test_velocity_depends_on_where_the_signal_sits_once_perturbed():
    """The y tokens carry a learned position: moving a feature along the window matters."""
    net, (x, t, _, f) = perturbed(flow()), inputs()
    y = jnp.zeros((BINS, C)).at[2].set(5.0)
    differs(run(net, x, t, y, f), run(net, x, t, jnp.roll(y, 20, axis=0), f))


def test_data_tokens_know_their_frequency():
    """Identical tokens at different frequencies enter the data stream as different values."""
    net = flow()
    values = net.encode(jnp.zeros((BINS, C)), jnp.asarray(F))[1][1][0]
    differs(values[0], values[1])
    moved = net.encode(jnp.zeros((BINS, C)), jnp.asarray(2 * F))[1][1][0]
    differs(values, moved)


def test_velocity_depends_on_f_once_perturbed():
    """f is the window's centre frequency: the flow must know which band it is looking at"""
    net, (x, t, y, _) = perturbed(flow()), inputs()
    differs(run(net, x, t, y, jnp.asarray(1.0e-3)), run(net, x, t, y, jnp.asarray(9.0e-3)))


def test_permutation_equivariant_once_perturbed():
    """Under a fixed encoding, reordering the slots' z and frame together reorders the output, and nothing else."""
    net = perturbed(flow())
    z, t, y, f = inputs()
    _, cache = net.encode(y, f)
    mu, log_sigma = rand((S, X), 21), 0.1 * rand((S, X), 22)
    perm = jnp.array([2, 0, 3, 1])
    permuted = net(z[perm], t, (mu[perm], log_sigma[perm]), f, cache)
    close(permuted, net(z, t, (mu, log_sigma), f, cache)[perm], atol=2e-3)


def test_output_is_differentiable_in_z():
    net, (x, t, y, f) = perturbed(flow()), inputs()
    g = jax.jacobian(lambda z: run(net, z, t, y, f))(x)
    assert g.shape == (S, 2 * X, S, X) and bool(jnp.all(jnp.isfinite(g)))


def test_one_block_is_the_least_a_one_way_backbone_takes():
    assert run(flow(blocks=1), *inputs()).shape == (S, 2 * X)
    with pytest.raises(AssertionError, match="needs a block"):
        flow(blocks=0)


def test_bf16_backbone_keeps_float32_params_and_output():
    net = flow(dtype=jnp.bfloat16)
    assert run(perturbed(net), *inputs()).dtype == jnp.float32
    assert all(
        p.dtype == jnp.float32 for p in jax.tree.leaves(eqx.filter(net, eqx.is_array))
    )


def test_no_biases_inside_the_blocks():
    assert flow().backbone.blocks.mlp_x.linear1.bias is None
    assert flow().backbone.blocks.mod_x_attn.linear.bias is None


def test_rejects_a_window_the_patch_does_not_divide():
    with pytest.raises(AssertionError, match="not divisible by 16"):
        flow(bins=56)

