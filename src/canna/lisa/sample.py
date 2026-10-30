"""Posterior sampling: the data's frame, then base draws transported toward the flow's z1 in it."""

import equinox as eqx
import jax
import jax.numpy as jnp
from einops import rearrange
from jaxtyping import Array

from .network import LisaFlow, reference_frame
from .problem import LisaGB

# the field is stiff near t=1: the narrow band's loudest f0 width stops shrinking by 32 steps
ODE_STEPS = 32
# draws pushed through the network at once: all 1024 on B need a 27 GiB buffer
CHUNK = 256


@eqx.filter_jit
def sample_posterior(
    problem: LisaGB,
    flow: LisaFlow,
    z: Array,
    y: Array,
    f: Array,
    ode_steps: int = ODE_STEPS,
    chunk: int = CHUNK,
) -> Array:
    """Midpoint steps of standard normal draws z along (m - z) / (1 - t) toward the flow's mean z1 m, then x = mu + sigma z."""
    # the data stream sees only (y, f): one encoding serves every draw and every step
    delta, cache = flow.encode(y, f)
    frame = reference_frame(problem, delta, f)

    @eqx.filter_vmap(in_axes=(None, 0))
    def push(flow: LisaFlow, u: Array) -> Array:
        dt = jnp.asarray(1.0 / ode_steps, u.dtype)

        def velocity(u: Array, t: Array) -> Array:
            m, _ = jnp.split(flow(u, t, frame, f, cache), 2, axis=-1)
            return (m - u) / (1 - t)

        def step(i: Array, u: Array) -> Array:
            t = i * dt
            return u + dt * velocity(u + dt / 2 * velocity(u, t), t + dt / 2)

        # fori_loop, not a python loop: unrolled, compile time grows with ode_steps
        return jax.lax.fori_loop(0, ode_steps, step, u)

    n = z.shape[0]
    if n <= chunk:
        z1 = push(flow, z)
    else:
        # each draw's path is its own, so chunking changes only the memory
        assert n % chunk == 0, f"{n} draws do not split into chunks of {chunk}"
        chunks = jax.lax.map(
            lambda c: push(flow, c), rearrange(z, "(c n) ... -> c n ...", n=chunk)
        )
        z1 = rearrange(chunks, "c n ... -> (c n) ...")
    mu, log_sigma = frame
    return mu + jnp.exp(log_sigma) * z1
