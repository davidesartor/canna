"""Loss of XS-fd along x -> z, and of x rescaled per layer to y's or z's norm (same samples)."""
# Loads XS-fd's checkpoint, so run it on the code that wrote it: a worktree at 7adfd3d,
# PYTHONPATH=<worktree>/src .venv/bin/python lisa_checks/<this file> (F36).
import sys, warnings; warnings.filterwarnings("ignore")
from functools import partial
import jax, jax.numpy as jnp, jax.random as jr, numpy as np, equinox as eqx, optax
sys.argv = ["x", "--config", "XS-fd"]
import canna.lisa.train as T
args = T.parse_args()
state, epoch, _ = T.TrainState.from_config(args).restore_from(T.checkpoint_manager(T.run_dir(args)))
sf = [s for s in state.opt_state if isinstance(s, optax.contrib.ScheduleFreeState)][0]
params, static = eqx.partition(state.flow, eqx.is_inexact_array)
y, z = params, sf.z
x = jax.tree.map(lambda a, b: (a - 0.1 * b) / 0.9, y, z)
batches = [jax.vmap(partial(T.train_sample, state.problem, time_power=args.time_power))(jr.split(jr.key(10_000 + i), 256)) for i in range(8)]
@eqx.filter_jit
def loss(p, b):
    return T.flow_loss(eqx.combine(p, static), b)
L = lambda p: float(np.mean([loss(p, b) for b in batches]))
line = lambda a: jax.tree.map(lambda xi, zi: xi + a * (zi - xi), x, z)
print("along x -> z (alpha = 0 is x, 0.111 is y, 1 is z):")
for a in (0.0, 0.05, 0.111, 0.2, 0.3, 0.5, 0.75, 1.0):
    print(f"  alpha {a:5.3f}: loss {L(line(a)):.4f}")
def rescaled(src, ref):
    return jax.tree.map(lambda s, r: s * (jnp.linalg.norm(r) / (jnp.linalg.norm(s) + 1e-30)) if s.size > 1e4 else s, src, ref)
print(f"x rescaled per layer to y's norms: {L(rescaled(x, y)):.4f}")
print(f"z rescaled per layer to y's norms: {L(rescaled(z, y)):.4f}")
print(f"y rescaled per layer to x's norms: {L(rescaled(y, x)):.4f}")
