"""Is optax's schedule-free averaging accurate in float32 over a million steps?

optax.contrib.schedule_free stores y (the params) and z in the param dtype and rebuilds x each
step as (y - (1 - b1) z) / b1, then x <- (1 - c) x + c z with c = w_t / sum(w), ~1/t once the
warmup is over. At t ~ 1e6 the increment c (z - x) is a few float32 ulps of the weight, and
(1 - c) itself rounds to a float32 value that stays fixed for ~1e5 steps. This replays that
recursion, op for op, in float32 and in float64 on the same z trajectory (a random walk with a
drift, steps of Muon's RMS-matched size), and reports how far the float32 average lands from
the float64 one, against how far the average itself moved.

    .venv/bin/python lisa_checks/sf_precision.py [steps] [warmup]
"""
import sys
import jax
import jax.numpy as jnp
import jax.random as jr

jax.config.update("jax_enable_x64", True)
steps = int(sys.argv[1]) if len(sys.argv) > 1 else 1_200_000
warmup = int(sys.argv[2]) if len(sys.argv) > 2 else 24_000
n, b1, lr, power = 512, 0.9, 1e-3, 2.0
step_rms = 0.2 * lr  # Muon's update RMS with consistent_rms=0.2


def run(dtype):
    key = jr.key(0)
    w0 = (0.04 * jr.normal(key, (n,))).astype(dtype)
    drift = jr.normal(jr.key(1), (n,))

    def body(carry, t):
        y, z, wsum, maxlr = carry
        k = jr.fold_in(jr.key(2), t)
        lr_t = lr * jnp.minimum(1.0, (t + 1.0) / warmup)
        # the base update: a noisy step around a slow drift, the same in both precisions
        u = step_rms * (0.3 * drift + jr.normal(k, (n,))) * (lr_t / lr)
        maxlr = jnp.maximum(maxlr, lr_t).astype(dtype)
        w = maxlr ** power
        wsum = (wsum + w).astype(dtype)
        c = (w / wsum).astype(dtype)
        z_new = (z + u.astype(dtype)).astype(dtype)
        x = ((y - (1.0 - b1) * z) / b1).astype(dtype)
        x = ((1.0 - c) * x + c * z_new).astype(dtype)
        y_new = (b1 * x + (1.0 - b1) * z_new).astype(dtype)
        y = (y + (y_new - y)).astype(dtype)  # optax returns updates, apply_updates adds them
        return (y, z_new, wsum, maxlr), None

    zero = jnp.zeros((), dtype)
    (y, z, _, _), _ = jax.lax.scan(jax.jit(body), (w0, w0, zero, zero), jnp.arange(steps, dtype=jnp.float64))
    x = (y - (1.0 - b1) * z) / b1
    return w0, x.astype(jnp.float64), z.astype(jnp.float64)


w0, x64, z64 = run(jnp.float64)
_, x32, z32 = run(jnp.float32)
moved = jnp.linalg.norm(x64 - w0)
print(f"{steps} steps: the average moved |x - x0| = {moved:.3e}, z ended |z - x| = {jnp.linalg.norm(z64 - x64):.3e}")
print(f"float32 vs float64: |x32 - x64| = {jnp.linalg.norm(x32 - x64):.3e}  ({jnp.linalg.norm(x32 - x64) / moved:.1%} of the move)")
print(f"  component along x64 (a rescaling): {jnp.dot(x32 - x64, x64) / jnp.dot(x64, x64):+.2e} of |x|")
print(f"  z: |z32 - z64| = {jnp.linalg.norm(z32 - z64):.3e}")
