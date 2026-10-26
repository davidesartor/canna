"""Posterior sampling: transport base draws along the learned velocity field."""

import equinox as eqx
import jax
import jax.numpy as jnp
from einops import rearrange
from jaxtyping import Array

from .network import LisaFlow

# the field is stiff near t=1: the narrow band's loudest f0 width stops shrinking by 32 steps
ODE_STEPS = 32
# draws pushed through the network at once: all 1024 on B need a 27 GiB buffer
CHUNK = 256


@eqx.filter_jit
def sample_posterior(
    flow: LisaFlow,
    u: Array,
    y: Array,
    f: Array,
    ode_steps: int = ODE_STEPS,
    time_power: float = 1.0,
    chunk: int = CHUNK,
) -> Array:
    """RK4 transport of base draws u along the learned velocity d/ds, scaled by ds/dt.

    An x- or residual-predicting flow takes midpoint steps toward its x1 instead, exact for a constant x1.
    """
    # the data stream sees only (y, f): one encoding serves every draw and every step
    cache = flow.encode(y, f)
    remaining = lambda t: (1 - t) ** time_power

    @eqx.filter_vmap(in_axes=(None, 0))
    def push(flow: LisaFlow, u: Array) -> Array:
        dt = jnp.asarray(1.0 / ode_steps, u.dtype)

        # the flow's output and its x1 (a velocity's endpoint), which a self-conditioned flow is fed back
        def predict(u: Array, t: Array, x_hat: Array | None) -> tuple[Array, Array]:
            out = flow(u, t, y, f, x_hat if flow.self_condition else None, cache)
            if flow.prediction == "x":
                return out, out
            if flow.prediction == "velocity":
                return out, u + remaining(t) * out
            z, log_var = jnp.split(out, 2, axis=-1)
            return out, u + jnp.exp(log_var / 2) * z

        def velocity(u: Array, t: Array, x_hat: Array) -> tuple[Array, Array]:
            v, x1 = predict(u, t, x_hat)
            v = v if time_power == 1 else time_power * (1 - t) ** (time_power - 1) * v
            return v, x1.astype(x_hat.dtype)

        def rk4_step(i: Array, carry: tuple[Array, Array]) -> tuple[Array, Array]:
            u, h = carry
            t = i * dt
            k1, h = velocity(u, t, h)
            k2, h = velocity(u + k1 * dt / 2, t + dt / 2, h)
            k3, h = velocity(u + k2 * dt / 2, t + dt / 2, h)
            k4, h = velocity(u + k3 * dt, t + dt, h)
            return u + (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6, h

        # the share of the remaining path to x1 that s covers from t to t_next
        toward = lambda u, x1, t, t_next: u + (
            1 - remaining(t_next) / remaining(t)
        ) * (x1 - u)

        def x_step(i: Array, carry: tuple[Array, Array]) -> tuple[Array, Array]:
            u, h = carry
            t = i * dt
            h = predict(u, t, h)[1].astype(h.dtype)
            h = predict(toward(u, h, t, t + dt / 2), t + dt / 2, h)[1].astype(h.dtype)
            return toward(u, h, t, t + dt), h

        # the carried x1 takes the dtype of the flow's x1: float32 for x, float64 (the draws') for an anchored x1
        first = lambda: predict(u, jnp.zeros((), u.dtype), None)[1]
        shape = jax.eval_shape(first)
        x_hat = first() if flow.self_condition else jnp.zeros(shape.shape, shape.dtype)
        # fori_loop, not a python loop: unrolled, compile time grows with ode_steps
        step = rk4_step if flow.prediction == "velocity" else x_step
        return jax.lax.fori_loop(0, ode_steps, step, (u, x_hat))[0]

    n = u.shape[0]
    if n <= chunk:
        return push(flow, u)
    # each draw's path is its own, so chunking changes only the memory
    assert n % chunk == 0, f"{n} draws do not split into chunks of {chunk}"
    chunks = jax.lax.map(
        lambda c: push(flow, c), rearrange(u, "(c n) ... -> c n ...", n=chunk)
    )
    return rearrange(chunks, "c n ... -> (c n) ...")
