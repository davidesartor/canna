"""MMDiTFlow contracts at the default config, over the shared module fixtures."""

import jax.numpy as jnp
import numpy as np
import pytest

from ._helpers import close, differs, rand


def test_mmditflow_output_triple_shapes(flow):
    net = flow
    x, y, t = rand((4, 2), 1), rand((8, 8, 3), 2), jnp.asarray(0.3)
    dx, xt, yt = net(x, y, t)
    assert dx.shape == (4, 2)
    assert xt.shape == (4, 2)
    assert yt.shape == (8, 8, 3)


def test_mmditflow_batch_ranks(flow):
    net = flow
    for lead in [(), (5,), (2, 5)]:
        dx, xt, yt = net(rand(lead + (4, 2), 1), rand(lead + (8, 8, 3), 2), rand(lead))
        assert dx.shape == lead + (4, 2)
        assert xt.shape == lead + (4, 2)
        assert yt.shape == lead + (8, 8, 3)


def test_mmditflow_scalar_time_broadcasts_over_batch(flow):
    net = flow
    x, y = rand((3, 4, 2), 1), rand((3, 8, 8, 3), 2)
    a = net(x, y, jnp.asarray(0.25))[0]
    b = net(x, y, jnp.full((3,), 0.25))[0]
    close(a, b)


def test_mmditflow_y_target_depends_on_y(flow):
    net = flow
    x, t = rand((4, 2), 1), jnp.asarray(0.3)
    a = net(x, rand((8, 8, 3), 2), t)[2]
    b = net(x, rand((8, 8, 3), 12), t)[2]
    assert not np.allclose(np.asarray(a), np.asarray(b), atol=1e-5)


def test_mmditflow_heads_are_distinct(flow):
    net = flow
    dx, xt, _ = net(rand((4, 2), 1), rand((8, 8, 3), 2), jnp.asarray(0.3))
    assert not np.allclose(np.asarray(dx), np.asarray(xt), atol=1e-5)


def test_mmditflow_source_count_is_free(flow):
    net = flow
    y, t = rand((8, 8, 3), 2), jnp.asarray(0.3)
    for n in [1, 2, 7]:
        dx, xt, _ = net(rand((n, 2), 1), y, t)
        assert dx.shape == (n, 2) and xt.shape == (n, 2)


def test_mmditflow_rejects_indivisible_image_resolution(flow):
    net = flow
    with pytest.raises(Exception):
        net(rand((4, 2), 1), rand((6, 8, 3), 2), jnp.asarray(0.3))


def test_mmditflow_finite_at_time_endpoints(flow):
    net = flow
    for t in [0.0, 1.0]:
        outs = net(rand((4, 2), 1), rand((8, 8, 3), 2), jnp.asarray(t))
        assert all(bool(jnp.all(jnp.isfinite(o))) for o in outs)


def test_mmditflow_ignores_conditioning_at_init(flow):
    net = flow
    x = rand((4, 2), 1)
    a = net(x, rand((8, 8, 3), 2), jnp.asarray(0.1))[0]
    b = net(x, 10.0 * rand((8, 8, 3), 9), jnp.asarray(0.9))[0]
    close(a, b, atol=0.0)


def test_mmditflow_velocity_depends_on_y_with_perturbed_params(perturbed_flow):
    net = perturbed_flow
    x, t = rand((4, 2), 1), jnp.asarray(0.3)
    differs(net(x, rand((8, 8, 3), 2), t)[0], net(x, rand((8, 8, 3), 12), t)[0])


def test_mmditflow_velocity_depends_on_t_with_perturbed_params(perturbed_flow):
    net = perturbed_flow
    x, y = rand((4, 2), 1), rand((8, 8, 3), 2)
    differs(net(x, y, jnp.asarray(0.1))[0], net(x, y, jnp.asarray(0.9))[0])


def test_mmditflow_permutation_equivariant_with_perturbed_params(perturbed_flow):
    net = perturbed_flow
    x, y, t = rand((4, 2), 1), rand((8, 8, 3), 2), jnp.asarray(0.3)
    perm = jnp.asarray([3, 1, 0, 2])
    dx, xt, yt = net(x, y, t)
    pdx, pxt, pyt = net(x[perm], y, t)
    close(pdx, dx[perm])
    close(pxt, xt[perm])
    close(pyt, yt)


def test_mmditflow_y_target_depends_on_x_with_perturbed_params(perturbed_flow):
    net = perturbed_flow
    y, t = rand((8, 8, 3), 2), jnp.asarray(0.3)
    differs(net(rand((4, 2), 1), y, t)[2], net(rand((4, 2), 8), y, t)[2])


def test_mmditflow_y_target_ignores_x_at_init(flow):
    net = flow
    y, t = rand((8, 8, 3), 2), jnp.asarray(0.3)
    close(net(rand((4, 2), 1), y, t)[2], net(rand((4, 2), 8), y, t)[2], atol=0.0)


def test_mmditflow_batch_independent_with_perturbed_params(perturbed_flow):
    net = perturbed_flow
    x, y, t = rand((3, 4, 2), 1), rand((3, 8, 8, 3), 2), rand((3,), 3)
    dx = net(x, y, t)[0]
    close(dx, jnp.stack([net(x[i], y[i], t[i])[0] for i in range(3)]))


def test_mmditflow_perturbed_outputs_stay_finite(perturbed_flow):
    net = perturbed_flow
    outs = net(rand((4, 2), 1), rand((8, 8, 3), 2), jnp.asarray(0.3))
    assert all(bool(jnp.all(jnp.isfinite(o))) for o in outs)
