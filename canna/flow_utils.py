from typing import Callable, Optional
from jaxtyping import Array, Float, Scalar, Key
import jax
import jax.numpy as jnp
import jax.random as jr
from flax import nnx
import optax

from . import networks


@nnx.jit(static_argnames=("ode_steps", "exponential_map"))
def sample_posterior(
    flow: networks.MMDiT,
    u: Float[Array, "N S D"],
    y: Float[Array, "T F C"],
    ode_steps: int = 4,
    exponential_map: Optional[Callable] = None,
) -> Float[Array, "N S D"]:
    """RK4-integrate ``flow``'s flow velocity t=0->1 to draw a batch of posterior samples for one observation ``y``."""

    @nnx.vmap(in_axes=(None, 0))
    def push(flow, u: Float[Array, "S D"]) -> Float[Array, "S D"]:
        # unroll the fixed (static) ode_steps in Python: an nnx module inside
        # jax.lax.fori_loop trips the Param trace-level check (unlike eqx pytrees)
        dt = 1 / ode_steps
        for i in range(ode_steps):
            # runge-kutta 4th order integration of the flow velocity field
            t = jnp.asarray(i * dt, u.dtype)
            k1, _, _ = flow(u, y, t)
            k2, _, _ = flow(u + k1 * dt / 2, y, t + dt / 2)
            k3, _, _ = flow(u + k2 * dt / 2, y, t + dt / 2)
            k4, _, _ = flow(u + k3 * dt, y, t + dt)
            du = (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6
            u = (u + du) if exponential_map is None else exponential_map(u, du)
        return u

    return push(flow, u)


def flow_train_sample(
    key: Key,
    u1: Float[Array, "S D"],
    geodesic: Optional[Callable] = None,
    match_sources: Optional[Callable] = None,
) -> tuple[Float[Array, "S D"], Float[Array, "S D"], Scalar]:
    """Given a single target ``u1``, build the flow-matching triple ``(ut, du, t)``. vmap for a batch."""
    key_u0, key_t = jr.split(key)
    # base point from the prior, coupled to the target
    u0 = jr.uniform(key_u0, shape=u1.shape, dtype=u1.dtype)
    if match_sources is not None:
        u0, _ = match_sources(u0, u1)

    # ode time and flow velocity
    t = jr.uniform(key_t, minval=0.0, maxval=1.0, dtype=u1.dtype)
    if geodesic is None:  # default to Euclidean geometry
        ut = u0 + t * (u1 - u0)
        du = u1 - u0
    else:
        ut = geodesic(t, u0, u1)
        du = jax.jacobian(geodesic)(t, u0, u1)
    return ut, du, t


def loss_flow_matching(
    du_pred: Float[Array, "B S D"], du: Float[Array, "B S D"]
) -> tuple[Scalar, Scalar]:
    """Flow-matching velocity MSE and its target variance."""
    return jnp.mean(optax.l2_loss(du_pred, du)), jnp.var(du)


def loss_param_regression(
    u1_pred: Float[Array, "... S D"],
    u_targ: Float[Array, "... S D"],
    match_sources: Optional[Callable] = None,
) -> tuple[Scalar, Scalar]:
    """Auxiliary MLE regression MSE and its target variance (sources aligned to targets first)."""
    if match_sources is not None:
        u1_pred, _ = jax.vmap(match_sources)(u1_pred, u_targ)
    return jnp.mean(optax.l2_loss(u1_pred, u_targ)), jnp.var(u_targ)


def loss_signal_regression(
    y_recon: Float[Array, "... T F C"], y_targ: Float[Array, "... T F C"]
) -> tuple[Scalar, Scalar]:
    """Auxiliary WDM reconstruction MSE and its target variance (crop both to their common grid)."""
    t = min(y_recon.shape[-3], y_targ.shape[-3])
    f = min(y_recon.shape[-2], y_targ.shape[-2])
    y_recon, y_targ = y_recon[..., :t, :f, :], y_targ[..., :t, :f, :]
    return jnp.mean(optax.l2_loss(y_recon, y_targ)), jnp.var(y_targ)


def reweight_losses(
    losses: Float[Array, "3"],
    weight: Scalar,
    variance_ema: Float[Array, "3"],
    eps: float = 1e-8,
) -> Scalar:
    """Blend per-term losses: normalize each by its EMA target variance, weight flow vs auxiliaries by ``weight``."""
    weights = jnp.array([weight, 1.0 - weight, 1.0 - weight])
    weights = weights / jax.lax.stop_gradient(variance_ema + eps)
    return jnp.sum(losses * weights)
