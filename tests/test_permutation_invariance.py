"""Permutation equivariance of the MMDiT flow on the real LISA problem, untrained.

The x stream carries no positional embedding, so relabelling the sources must commute
with the network: integrating then permuting == permuting then integrating. Checked
through a full RK4 flow integration, not a single forward pass, so a leak anywhere in
the composed trajectory shows up.
"""

import itertools

import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import pytest
from flax import nnx

from canna import lisa, networks

from _bench import CONFIG_BY_NAME, wdm_shape
from _helpers import problem_name, save_text

CONFIG = CONFIG_BY_NAME["B"]  # train.py default
N_POINTS = 3
ODE_STEPS = 4
SOURCE_COUNTS = [2, 3]
GATE_SCALE = 0.1

TRAIN_STEPS = 3
TRAIN_BATCH = 2
LEARNING_RATE = 1e-3

# float32 is what train.py runs; float64 pins the property itself, clear of roundoff
DTYPES = [(jnp.float32, 1e-5), (jnp.float64, 1e-12)]


def build_active_model(dtype, seed: int = 0) -> networks.MMDiT:
    """MMDiT at real LISA dims with the adaLN gates randomized instead of zero-init.

    Zero-init gates make every block the identity, which would leave the network
    pointwise in x and so equivariant for trivial reasons -- randomizing them turns
    the attention on so the check actually exercises token mixing.
    """
    model = networks.MMDiT(
        x_dim=len(lisa.PARAMETER_NAMES),
        y_channels=len(lisa.CHANNEL_NAMES),
        hidden_dim=CONFIG.hidden_dim,
        num_blocks=CONFIG.num_blocks,
        num_heads=CONFIG.num_heads,
        dtype=dtype,
        param_dtype=dtype,
        rngs=nnx.Rngs(seed),
    )

    graphdef, params = nnx.split(model, nnx.Param)
    pure = nnx.to_pure_dict(params)

    def activate(path, leaf):
        name = "/".join(str(getattr(k, "key", getattr(k, "idx", k))) for k in path)
        if "mod" not in name:
            return leaf
        key = jr.fold_in(jr.key(seed), hash(name) % (2**31))
        return leaf + GATE_SCALE * jr.normal(key, leaf.shape, leaf.dtype)

    nnx.replace_by_pure_dict(params, jax.tree_util.tree_map_with_path(activate, pure))
    return nnx.merge(graphdef, params)


def train_briefly(model, dtype, n_sources: int, seed: int = 0):
    """A few optimizer steps on random targets, purely to move the weights off init.

    Not learning anything -- the point is that equivariance is structural and must
    survive arbitrary weight values, so the check should not run only at init.
    """
    optimizer = nnx.Optimizer(model, tx=optax.adamw(LEARNING_RATE), wrt=nnx.Param)

    @nnx.jit
    def step(model, optimizer, x, y, t, du):
        def loss_fn(model):
            dx, _, _ = model(x, y, t)
            return jnp.mean(optax.l2_loss(dx, du))

        optimizer.update(model, nnx.grad(loss_fn)(model))

    x_dim = len(lisa.PARAMETER_NAMES)
    for i in range(TRAIN_STEPS):
        keys = jr.split(jr.fold_in(jr.key(seed), i), 4)
        step(
            model,
            optimizer,
            jr.uniform(keys[0], (TRAIN_BATCH, n_sources, x_dim), dtype),
            jr.normal(keys[1], (TRAIN_BATCH, *wdm_shape()), dtype),
            jr.uniform(keys[2], (TRAIN_BATCH,), dtype),
            jr.normal(keys[3], (TRAIN_BATCH, n_sources, x_dim), dtype),
        )
    return model


def integrate(model, x, y, ode_steps: int = ODE_STEPS):
    """RK4-integrate the flow t: 0->1, same scheme as scripts/eval.py."""
    dt = jnp.asarray(1.0 / ode_steps, x.dtype)
    for i in range(ode_steps):
        t = jnp.full(x.shape[:-2], i * dt, x.dtype)
        k1, _, _ = model(x, y, t)
        k2, _, _ = model(x + k1 * dt / 2, y, t + dt / 2)
        k3, _, _ = model(x + k2 * dt / 2, y, t + dt / 2)
        k4, _, _ = model(x + k3 * dt, y, t + dt)
        x = lisa.exponential_map(x, (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6)
    return x


@pytest.fixture(scope="module")
def report_lines():
    lines = [
        f"{problem_name()}  config={CONFIG.name}  x_dim={len(lisa.PARAMETER_NAMES)}  "
        f"y={wdm_shape()}  points={N_POINTS}  rk4_steps={ODE_STEPS}  "
        f"train={TRAIN_STEPS} steps @ batch {TRAIN_BATCH}",
        f"{'dtype':>9} {'sources':>8} {'state':>10} {'permutation':>14} "
        f"{'max |A - B|':>13} {'tol':>9}",
    ]
    yield lines
    save_text("permutation_invariance", "\n".join(lines))


@pytest.mark.parametrize("trained", [False, True], ids=["init", "trained"])
@pytest.mark.parametrize("dtype,tol", DTYPES, ids=lambda v: getattr(v, "__name__", v))
@pytest.mark.parametrize("n_sources", SOURCE_COUNTS)
def test_integration_commutes_with_permutation(
    dtype, tol, n_sources, trained, report_lines
):
    model = build_active_model(dtype)
    if trained:
        model = train_briefly(model, dtype, n_sources)
    x = jr.uniform(jr.key(0), (N_POINTS, n_sources, len(lisa.PARAMETER_NAMES)), dtype)
    y = jr.normal(jr.key(1), (N_POINTS, *wdm_shape()), dtype)

    pushed = integrate(model, x, y)

    # the gates must be live, or a pointwise network would pass this for free
    spread = float(jnp.max(jnp.abs(pushed - x)))
    assert spread > 1e-6, f"flow is ~identity (max move {spread:.2e}); gates not active"

    for sigma in itertools.permutations(range(n_sources)):
        perm = jnp.asarray(sigma)
        push_then_perm = pushed[:, perm]
        perm_then_push = integrate(model, x[:, perm], y)
        err = float(jnp.max(jnp.abs(push_then_perm - perm_then_push)))
        report_lines.append(
            f"{jnp.dtype(dtype).name:>9} {n_sources:>8} "
            f"{'trained' if trained else 'init':>10} {str(sigma):>14} "
            f"{err:>13.3e} {tol:>9.0e}"
        )
        assert err < tol, (
            f"permutation {sigma} does not commute with the flow: "
            f"max |A - B| = {err:.3e} > {tol:.0e}"
        )
