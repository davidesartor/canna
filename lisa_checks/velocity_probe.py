"""What the network moves: per flow coordinate, the velocity it predicts against the target (F47).

On fresh training samples, for the run's averaged weights x (or y with --weights training):
- the spread (std) of the predicted velocity over the target's, per coordinate: a coordinate
  the network never moves (phi0 on every run to F46) sits near 0;
- R^2 of the predicted velocity, per coordinate and band of path position s;
- the reconstruction head's phase error against x_t's own, i.e. whether the head learns phi0
  or copies it.

    .venv/bin/python lisa_checks/velocity_probe.py B-fd-rope outputs/18-lisa-B-rope-parent [flags]
It also prints the velocity head's last-layer row norms (F48).
The second argument is an --output_dir holding lisa-<config> (a symlink to the run folder will do).
"""

import sys
import warnings
from functools import partial

warnings.filterwarnings("ignore")
import equinox as eqx
import jax
import jax.random as jr
import numpy as np

from canna.lisa import train as T

config, out = sys.argv[1], sys.argv[2]
sys.argv = ["velocity_probe", "--config", config, "--output_dir", out, *sys.argv[3:]]
args = T.parse_args()
state, epoch, _ = T.load_trained(args)
problem, flow = state.problem, state.flow
names = problem.flow_names  # the last two are phi0, or phi- with polarisation: helicity
sample = jax.jit(jax.vmap(partial(T.train_sample, problem, time_power=args.time_power,
                                  templates=flow.templates)))
both = eqx.filter_jit(lambda f, b: jax.vmap(f)(b.xt, b.t, b.y, b.f))
P, X, D, XT, TG, S = [], [], [], [], [], []
for k in range(16):
    b = sample(jr.split(jr.key(200 + k), 64))
    dv, x1 = both(flow, b)
    for lst, v in zip((P, X, D, XT, TG), (dv, x1, b.dx, b.xt, b.x_target)):
        lst.append(np.asarray(v))
    S.append(np.asarray(T.path_position(b.t, args.time_power)))
P, X, D, XT, TG, S = map(np.concatenate, (P, X, D, XT, TG, S))

print(f"{config} epoch {epoch} ({args.weights} weights), {len(S)} samples")
print("coordinate          " + " ".join(f"{n:>8s}" for n in names))
print("predicted / target  " + " ".join(f"{v:8.3f}" for v in P.reshape(-1, 11).std(0) / D.reshape(-1, 11).std(0)))
ang = lambda v: np.arctan2(v[..., 10], v[..., 9])
wrap = lambda a: (a + np.pi) % (2 * np.pi) - np.pi
for lo, hi in [(0, 0.5), (0.5, 0.9), (0.9, 0.99), (0.99, 1.0)]:
    m = (S >= lo) & (S < hi)
    p, t = P[m].reshape(-1, 11), D[m].reshape(-1, 11)
    r2 = 1 - ((p - t) ** 2).mean(0) / t.var(0)
    head, xt = np.median(np.abs(wrap(ang(X[m]) - ang(TG[m])))), np.median(np.abs(wrap(ang(XT[m]) - ang(TG[m]))))
    print(f"R^2, s {lo:.2f}-{hi:.2f}     " + " ".join(f"{v:8.3f}" for v in r2)
          + f"   | phase error: head {head:.3f}, x_t {xt:.3f} rad")
# the velocity head's last-layer rows (AdamW), each over the median of the 11: F48 found the phi0
# rows at 0.88 in XS-fd and 0.08-0.14 once the decay reached them
rows = np.linalg.norm(np.asarray(flow.x_unembed.linear2.weight)[:11], axis=-1)
print("head row / median   " + " ".join(f"{v:8.3f}" for v in rows / np.median(rows)))
