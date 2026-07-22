"""MLPFlow contracts at the default config."""

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx


from ._helpers import close, differs, mlpflow, perturbed, rand


def test_mlpflow_output_triple_shapes():
    net = mlpflow()
    dx, xt, yt = net(rand((3,), 1), rand((5,), 2), jnp.asarray(0.3))
    assert dx.shape == (3,)
    assert xt.shape == (3,)
    assert yt.shape == (5,)


def test_mlpflow_batch_ranks():
    net = mlpflow()
    for lead in [(), (4,), (2, 4)]:
        dx, xt, yt = net(rand(lead + (3,), 1), rand(lead + (5,), 2), rand(lead))
        assert dx.shape == lead + (3,)
        assert xt.shape == lead + (3,)
        assert yt.shape == lead + (5,)


def test_mlpflow_batch_independent():
    net = mlpflow()
    x, y, t = rand((4, 3), 1), rand((4, 5), 2), rand((4,), 3)
    dx, xt, yt = net(x, y, t)
    parts = [net(x[i], y[i], t[i]) for i in range(4)]
    close(dx, jnp.stack([p[0] for p in parts]))
    close(xt, jnp.stack([p[1] for p in parts]))
    close(yt, jnp.stack([p[2] for p in parts]))


def test_mlpflow_scalar_time_broadcasts_over_batch():
    net = mlpflow()
    x, y = rand((4, 3), 1), rand((4, 5), 2)
    dx_scalar = net(x, y, jnp.asarray(0.25))[0]
    dx_vector = net(x, y, jnp.full((4,), 0.25))[0]
    close(dx_scalar, dx_vector)


def test_mlpflow_velocity_depends_on_x():
    net = mlpflow()
    y, t = rand((5,), 2), jnp.asarray(0.4)
    a = net(rand((3,), 1), y, t)[0]
    b = net(rand((3,), 8), y, t)[0]
    assert not np.allclose(np.asarray(a), np.asarray(b), atol=1e-5)


def test_mlpflow_heads_are_distinct():
    net = mlpflow()
    dx, xt, _ = net(rand((3,), 1), rand((5,), 2), jnp.asarray(0.4))
    assert not np.allclose(np.asarray(dx), np.asarray(xt), atol=1e-5)


def test_mlpflow_deterministic():
    net = mlpflow()
    args = (rand((4, 3), 1), rand((4, 5), 2), rand((4,), 3))
    a = net(*args)
    b = net(*args)
    for u, v in zip(a, b):
        close(u, v)


def test_mlpflow_finite_at_time_endpoints():
    net = mlpflow()
    for t in [0.0, 1.0]:
        outs = net(rand((3,), 1), rand((5,), 2), jnp.asarray(t))
        assert all(bool(jnp.all(jnp.isfinite(o))) for o in outs)


def test_mlpflow_velocity_is_differentiable_in_x():
    y, t, x_shape = (rand((5,), 2), jnp.asarray(0.4), (3,))
    graphdef, state = nnx.split(mlpflow())

    def velocity(state, z):
        return nnx.merge(graphdef, state)(z, y, t)[0]

    g = jax.jacobian(velocity, argnums=1)(state, rand(x_shape, 1))
    assert g.shape == x_shape + x_shape and bool(jnp.all(jnp.isfinite(g)))


def test_mlpflow_ignores_conditioning_at_init():
    net = mlpflow()
    x = rand((3,), 1)
    a = net(x, rand((5,), 2), jnp.asarray(0.1))[0]
    b = net(x, 10.0 * rand((5,), 9), jnp.asarray(0.9))[0]
    close(a, b, atol=0.0)


def test_mlpflow_velocity_depends_on_t_with_perturbed_params():
    net = perturbed(mlpflow())
    x, y = rand((3,), 1), rand((5,), 2)
    differs(net(x, y, jnp.asarray(0.1))[0], net(x, y, jnp.asarray(0.9))[0])


def test_mlpflow_velocity_depends_on_y_with_perturbed_params():
    net = perturbed(mlpflow())
    x, t = rand((3,), 1), jnp.asarray(0.4)
    differs(net(x, rand((5,), 2), t)[0], net(x, rand((5,), 7), t)[0])


def test_mlpflow_y_target_ignores_x_with_perturbed_params():
    net = perturbed(mlpflow())
    y, t = rand((5,), 2), jnp.asarray(0.4)
    close(net(rand((3,), 1), y, t)[2], net(rand((3,), 8), y, t)[2], atol=0.0)


def test_mlpflow_y_target_ignores_t_with_perturbed_params():
    net = perturbed(mlpflow())
    x, y = rand((3,), 1), rand((5,), 2)
    close(net(x, y, jnp.asarray(0.1))[2], net(x, y, jnp.asarray(0.9))[2], atol=0.0)


def test_mlpflow_perturbed_outputs_stay_finite():
    net = perturbed(mlpflow())
    outs = net(rand((4, 3), 1), rand((4, 5), 2), rand((4,), 3))
    assert all(bool(jnp.all(jnp.isfinite(o))) for o in outs)


def test_mlpflow_batch_independent_with_perturbed_params():
    net = perturbed(mlpflow())
    x, y, t = rand((4, 3), 1), rand((4, 5), 2), rand((4,), 3)
    dx = net(x, y, t)[0]
    close(dx, jnp.stack([net(x[i], y[i], t[i])[0] for i in range(4)]))
