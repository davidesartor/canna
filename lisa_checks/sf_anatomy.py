"""Anatomy of XS-fd's final schedule-free state: x (average), y (trained), z (fast sequence)."""
# Loads XS-fd's checkpoint, so run it on the code that wrote it: a worktree at 7adfd3d,
# PYTHONPATH=<worktree>/src .venv/bin/python lisa_checks/<this file> (F36).
import sys, warnings; warnings.filterwarnings("ignore")
import jax, numpy as np, equinox as eqx, optax
sys.argv = ["x", "--config", "XS-fd"]
import canna.lisa.train as T
args = T.parse_args()
state, epoch, _ = T.TrainState.from_config(args).restore_from(T.checkpoint_manager(T.run_dir(args)))
sf = [s for s in state.opt_state if isinstance(s, optax.contrib.ScheduleFreeState)][0]
init = T.TrainState.from_config(args).flow
P = lambda m: eqx.filter(m, eqx.is_inexact_array)
y, z, w0 = P(state.flow), sf.z, P(init)
x = jax.tree.map(lambda a, b: (a - 0.1 * b) / 0.9, y, z)
rows = []
for (path, xl), yl, zl, il in zip(jax.tree_util.tree_leaves_with_path(x), jax.tree.leaves(y), jax.tree.leaves(z), jax.tree.leaves(w0)):
    xl, yl, zl, il = (np.asarray(a, np.float64).ravel() for a in (xl, yl, zl, il))
    if xl.size < 1e5: continue
    nrm = np.linalg.norm
    cos = lambda a, b: a @ b / (nrm(a) * nrm(b) + 1e-30)
    rows.append((jax.tree_util.keystr(path), xl.size, nrm(il) / np.sqrt(il.size), nrm(xl) / np.sqrt(xl.size),
                 nrm(zl) / np.sqrt(zl.size), cos(xl, zl), nrm(zl - xl) / nrm(xl), cos(xl - il, zl - il)))
print(f"{'parameter':52s} {'n':>9s} {'rms init':>8s} {'rms x':>7s} {'rms z':>7s} {'cos(x,z)':>8s} {'|z-x|/|x|':>9s} {'cos(x-w0,z-w0)':>14s}")
for r in sorted(rows, key=lambda r: -r[1])[:14]:
    print(f"{r[0][:52]:52s} {r[1]:9d} {r[2]:8.4f} {r[3]:7.4f} {r[4]:7.4f} {r[5]:8.3f} {r[6]:9.3f} {r[7]:14.3f}")
n = np.array([r[1] for r in rows]); w = n / n.sum()
print("size-weighted: cos(x,z) %.3f  |z-x|/|x| %.3f  |z|/|x| %.3f" % ((w * [r[5] for r in rows]).sum(), (w * [r[6] for r in rows]).sum(), (w * np.array([r[4] / r[3] for r in rows])).sum()))
print("SF state: step", int(sf.step_count), "weight_sum", float(sf.weight_sum), "max_lr", float(sf.max_lr))
