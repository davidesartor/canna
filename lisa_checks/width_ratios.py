"""Per-source posterior widths of one run against another on the 40 eval sources (F51).

Both runs' eval_draws.npz hold draws of the same ten injections. Each source's draws are matched
to the truth (eval.match_sources) and folded (mcmc.fold: f0 in bins, psi/phi0 onto the truth's
copy, longitude wrapped around it); the robust widths (1.4826 MAD) of run B over run A are then
summarised per SNR band, median [16-84%], with the sky in ecliptic (lambda, sin beta).

    .venv/bin/python lisa_checks/width_ratios.py <problem config> <run A folder> <run B folder>
"""
import sys
import warnings

warnings.filterwarnings("ignore")
import jax
import numpy as np

from canna.lisa import mcmc as M
from canna.lisa.eval import eval_injections, match_sources

OBLIQUITY = np.deg2rad(23.4392911)


def ecliptic(p):
    lon, lat = p[..., 3], p[..., 4]
    x, y, z = np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)
    c, s = np.cos(OBLIQUITY), np.sin(OBLIQUITY)
    return np.arctan2(c * y + s * z, x), -s * y + c * z


config, run_a, run_b = sys.argv[1:4]
problem, seed = M.load_problem(config)
_, windows, *_ = eval_injections(problem, seed)
names = ["f0", "A", "lambda", "sin beta", "psi", "sin iota", "phi0"]


def widths(run):
    d = np.load(f"outputs/{run}/eval_draws.npz")
    out, snr = [], []
    for j in range(len(d["truth"])):
        t = d["truth"][j]
        snr.extend(np.asarray(problem.source_snrs(t, windows[j])))
        fl = match_sources(problem, d["draws"][j], t, windows[j])
        for s in range(t.shape[0]):
            a = M.fold(fl[:, s], t[s], problem.t_obs)
            lam, sinb = ecliptic(fl[:, s])
            lam0, _ = ecliptic(t[s])
            lam = lam0 + (lam - lam0 + np.pi) % (2 * np.pi) - np.pi
            cols = np.stack([a[:, 0], a[:, 2], lam, sinb, a[:, 5], a[:, 6], a[:, 7]], 1)
            out.append(M.robust_width(cols))
    return np.array(out), np.array(snr)


wa, snr = widths(run_a)
wb, _ = widths(run_b)
r = wb / wa
print(f"{run_b} / {run_a}: robust width per source, median [16-84%] by SNR band")
print(f"{'SNR':>10s} {'n':>3s} " + "".join(f"{n:>20s}" for n in names))
for lo, hi in [(0, 15), (15, 100), (100, 300), (300, np.inf)]:
    m = (snr >= lo) & (snr < hi)
    cells = [f"{np.median(r[m, c]):.2f} [{np.percentile(r[m, c], 16):.2f}-{np.percentile(r[m, c], 84):.2f}]"
             for c in range(len(names))]
    print(f"{f'{lo}-{hi:g}':>10s} {m.sum():3d} " + "".join(f"{x:>20s}" for x in cells))
