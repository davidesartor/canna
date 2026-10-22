"""Sky offsets per source, split by the Doppler modulation index (F56, F57).

The index 2 pi f0 (1 AU / c) cos(beta) (beta the ecliptic latitude) says how much of a source's
sky information is in the Doppler phase (Cornish & Larson 2002): above ~1 rad it dominates,
below it the antenna pattern does. For every eval source above an SNR cut, this prints the flow
median's distance from the truth in flow widths, |z|, for ecliptic lambda and sin beta, per index
band and run, and the number of sources with |z| > 1.5 in either.

    .venv/bin/python lisa_checks/sky_by_doppler.py <config> <run folder> [more run folders]
"""
import sys
import warnings

warnings.filterwarnings("ignore")
import jax
import numpy as np

from canna.lisa import mcmc as M
from canna.lisa.eval import eval_injections, match_sources

OBLIQUITY = np.deg2rad(23.4392911)
LIGHT_SECONDS_PER_AU = 499.004784
SNR_CUT = 15.0


def ecliptic(p):
    lon, lat = p[..., 3], p[..., 4]
    x, y, z = np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)
    c, s = np.cos(OBLIQUITY), np.sin(OBLIQUITY)
    return np.arctan2(c * y + s * z, x), -s * y + c * z


def width(x):
    return 1.4826 * np.median(np.abs(x - np.median(x)))


config, runs = sys.argv[1], sys.argv[2:]
problem, seed = M.load_problem(config)
_, windows, *_ = eval_injections(problem, seed)
for run in runs:
    d = np.load(f"outputs/{run}/eval_draws.npz")
    rows = []
    for j in range(len(d["truth"])):
        snr = np.asarray(problem.source_snrs(d["truth"][j], windows[j]))
        fl = match_sources(problem, d["draws"][j], d["truth"][j], windows[j])
        for s in range(d["truth"].shape[1]):
            t = d["truth"][j, s]
            lam, sinb = ecliptic(fl[:, s])
            lam0, sinb0 = ecliptic(t)
            lam = lam0 + (lam - lam0 + np.pi) % (2 * np.pi) - np.pi
            index = 2 * np.pi * t[0] * LIGHT_SECONDS_PER_AU * np.sqrt(max(1 - sinb0**2, 0.0))
            rows.append([snr[s], index, abs(np.median(lam) - lam0) / width(lam),
                         abs(np.median(sinb) - sinb0) / width(sinb)])
    r = np.array(rows)
    r = r[r[:, 0] >= SNR_CUT]
    print(f"{run}: sources with SNR >= {SNR_CUT:g}, |median - truth| in flow widths")
    for lo, hi in [(0, 1), (1, np.inf)]:
        m = (r[:, 1] >= lo) & (r[:, 1] < hi)
        far = np.sum((r[m, 2] > 1.5) | (r[m, 3] > 1.5))
        print(f"  Doppler index {lo}-{hi:g} rad: n {m.sum():2d}, median |z| lambda {np.median(r[m, 2]):.2f}"
              f" sin beta {np.median(r[m, 3]):.2f}, mean |z| {r[m, 2].mean():.2f} {r[m, 3].mean():.2f};"
              f" |z| > 1.5 in either: {far}")
