"""SinusoidFlow: shapes, patch divisibility, and permutation invariance over sources."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from canna.sinusoid import SinusoidFlow

S, X, H, W, C, HIDDEN, HEADS = 4, 2, 16, 16, 3, 16, 2


def rand(shape, seed: int = 0):
    return jax.random.normal(jax.random.key(seed), shape)


def close(a, b, atol=1e-4):
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=atol, rtol=1e-4)


def differs(a, b, atol=1e-5):
    assert not np.allclose(np.asarray(a), np.asarray(b), atol=atol)


def flow(
    sources=S,
    x_dim=X,
    height=H,
    width=W,
    y_dim=C,
    hidden=HIDDEN,
    heads=HEADS,
    blocks=2,
    stages=2,
    seed=0,
):
    return SinusoidFlow(
        (sources, x_dim),
        (height, width, y_dim),
        hidden,
        heads,
        blocks,
        stages,
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


def test_output_triple_shapes():
    dx, xt, yt = flow()(rand((S, X), 1), jnp.asarray(0.3), rand((H, W, C), 2))
    assert dx.shape == (S, X) and xt.shape == (S, X) and yt.shape == (H, W, C)


def test_takes_one_example_and_vmaps():
    dx, xt, yt = jax.vmap(flow())(
        rand((3, S, X), 1), rand((3,), 3), rand((3, H, W, C), 2)
    )
    assert dx.shape == (3, S, X)
    assert xt.shape == (3, S, X)
    assert yt.shape == (3, H, W, C)


def test_batch_rows_are_independent():
    net = perturbed(flow())
    x, t, y = rand((3, S, X), 1), rand((3,), 3), rand((3, H, W, C), 2)
    dx = jax.vmap(net)(x, t, y)[0]
    close(dx, jnp.stack([net(x[i], t[i], y[i])[0] for i in range(3)]))


@pytest.mark.parametrize("sources", [1, 2, 7])
def test_source_count_is_free(sources):
    net = flow()
    dx, xt, _ = net(rand((sources, X), 1), jnp.asarray(0.3), rand((H, W, C), 2))
    assert dx.shape == (sources, X) and xt.shape == (sources, X)


def test_velocity_and_x_target_are_distinct_heads():
    dx, xt, _ = flow()(rand((S, X), 1), jnp.asarray(0.3), rand((H, W, C), 2))
    differs(dx, xt)


@pytest.mark.parametrize("t", [0.0, 1.0])
def test_finite_at_the_time_endpoints(t):
    outs = flow()(rand((S, X), 1), jnp.asarray(t), rand((H, W, C), 2))
    assert all(bool(jnp.all(jnp.isfinite(o))) for o in outs)


def test_y_target_depends_on_y_at_init():
    net, x, t = flow(), rand((S, X), 1), jnp.asarray(0.3)
    differs(net(x, t, rand((H, W, C), 2))[2], net(x, t, rand((H, W, C), 12))[2])


def test_ignores_conditioning_at_init():
    net, x = flow(), rand((S, X), 1)
    a = net(x, jnp.asarray(0.1), rand((H, W, C), 2))[0]
    b = net(x, jnp.asarray(0.9), 10.0 * rand((H, W, C), 9))[0]
    close(a, b, atol=0.0)


def test_y_target_ignores_x_at_init():
    net, t, y = flow(), jnp.asarray(0.3), rand((H, W, C), 2)
    close(net(rand((S, X), 1), t, y)[2], net(rand((S, X), 8), t, y)[2], atol=0.0)


def test_velocity_depends_on_y_once_perturbed():
    net, x, t = perturbed(flow()), rand((S, X), 1), jnp.asarray(0.3)
    differs(net(x, t, rand((H, W, C), 2))[0], net(x, t, rand((H, W, C), 12))[0])


def test_velocity_depends_on_t_once_perturbed():
    net, x, y = perturbed(flow()), rand((S, X), 1), rand((H, W, C), 2)
    differs(net(x, jnp.asarray(0.1), y)[0], net(x, jnp.asarray(0.9), y)[0])


def test_y_target_depends_on_x_once_perturbed():
    """joint attention: unlike PointFlow, the y stream does see the sources"""
    net, t, y = perturbed(flow()), jnp.asarray(0.3), rand((H, W, C), 2)
    differs(net(rand((S, X), 1), t, y)[2], net(rand((S, X), 8), t, y)[2])


def test_permutation_equivariant_over_sources_once_perturbed():
    net = perturbed(flow())
    x, t, y = rand((S, X), 1), jnp.asarray(0.3), rand((H, W, C), 2)
    perm = jnp.asarray([3, 1, 0, 2])
    dx, xt, yt = net(x, t, y)
    perm_dx, perm_xt, perm_yt = net(x[perm], t, y)
    close(perm_dx, dx[perm])
    close(perm_xt, xt[perm])
    close(perm_yt, yt)


def test_x_stream_has_no_positional_embedding():
    net = perturbed(flow())
    token = rand((1, X), 1)
    x = jnp.concat([token, token + 1e-9], axis=0)
    dx = net(x, jnp.asarray(0.3), rand((H, W, C), 2))[0]
    close(dx[0], dx[1], atol=1e-5)


def test_velocity_is_differentiable_in_x():
    net, t, y = perturbed(flow()), jnp.asarray(0.3), rand((H, W, C), 2)
    g = jax.jacobian(lambda z: net(z, t, y)[0])(rand((S, X), 1))
    assert g.shape == (S, X, S, X) and bool(jnp.all(jnp.isfinite(g)))


def test_perturbed_outputs_stay_finite():
    outs = perturbed(flow())(rand((S, X), 1), jnp.asarray(0.3), rand((H, W, C), 2))
    assert all(bool(jnp.all(jnp.isfinite(o))) for o in outs)


def test_non_square_image():
    net = flow(width=32)
    dx, _, yt = net(rand((S, X), 1), jnp.asarray(0.3), rand((H, 32, C), 2))
    assert dx.shape == (S, X) and yt.shape == (H, 32, C)


def test_single_channel_image():
    net = flow(y_dim=1)
    dx, _, yt = net(rand((S, X), 1), jnp.asarray(0.3), rand((H, W, 1), 2))
    assert dx.shape == (S, X) and yt.shape == (H, W, 1)


def test_zero_blocks():
    net = flow(blocks=0)
    dx, xt, yt = net(rand((S, X), 1), jnp.asarray(0.3), rand((H, W, C), 2))
    assert dx.shape == (S, X) and xt.shape == (S, X) and yt.shape == (H, W, C)


@pytest.mark.parametrize("stages", [1, 2, 3])
def test_y_target_shape_survives_every_patch_stage_count(stages):
    net = flow(hidden=64, stages=stages)
    yt = net(rand((S, X), 1), jnp.asarray(0.3), rand((H, W, C), 2))[2]
    assert yt.shape == (H, W, C)


def test_rejects_an_image_the_patch_stages_do_not_divide():
    # NoisySinusoid's patch_downsample=4 is what patch_stages=2 is matched to;
    # a mismatch has to fail loudly at construction, not silently reshape
    with pytest.raises(AssertionError, match="not divisible by 4"):
        flow(height=6)
