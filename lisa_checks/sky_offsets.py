"""Are the flow's sky positions biased? Two readings of B runs' eval draws (F49).

1. Against the narrowband MCMC, per MCMC'd source, in the equatorial frame jaxgb uses and in
   ecliptic (lambda, sin beta): the flow's median minus the MCMC's, in MCMC widths and in the
   flow's own width; the angle between their mean directions; the truth's offset from the
   MCMC (the noise realisation); the sky shift the MCMC's own regression on f0 predicts from
   the flow's f0 offset; and the share of draws on the antipodal side / nearer the ecliptic
   mirror.
2. Over all 40 eval sources: the truth's rank in the flow's ecliptic marginals and the signed
   z = (flow median - truth) / flow width, by SNR band. A calibrated flow has mean z ~ 0.

    .venv/bin/python lisa_checks/sky_offsets.py 21-lisa-B-long 18-lisa-B-rope ...
(run folders under outputs/, all of the B problem)
"""
import sys
import warnings

warnings.filterwarnings("ignore")
import jax
import numpy as np

from canna.lisa import mcmc as M
from canna.lisa.eval import eval_injections, match_sources

OBLIQUITY = np.deg2rad(23.4392911)


def unit(lon, lat):
    return np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], -1)


def to_ecliptic(v):
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    c, s = np.cos(OBLIQUITY), np.sin(OBLIQUITY)
    return np.stack([x, c * y + s * z, -s * y + c * z], -1)


def lon_sinlat(v):
    return np.arctan2(v[..., 1], v[..., 0]), v[..., 2] / np.linalg.norm(v, axis=-1)


def wrap(a, centre):
    return centre + (a - centre + np.pi) % (2 * np.pi) - np.pi


def width(x):
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def angle(a, b):
    return np.arccos(np.clip(a @ b / np.linalg.norm(a) / np.linalg.norm(b), -1, 1))


problem, seed = M.load_problem("B-fd-rope-long")
_, windows, *_ = eval_injections(problem, seed)
truth0 = np.load(f"outputs/{sys.argv[1]}/eval_draws.npz")["truth"]
snr = np.asarray(jax.vmap(problem.source_snrs)(truth0, windows))
chains = M.load_chains(problem, seed, n=10**9)

for run in sys.argv[1:]:
    d = np.load(f"outputs/{run}/eval_draws.npz")
    draws, truth = d["draws"], d["truth"]
    print(f"\n===== {run}: against the MCMC")
    print("src   SNR    frame | med diff / MCMC w | med diff / flow w | angle / MCMC spread"
          " | truth-MCMC | from f0 offset")
    for (j, s), mc in sorted(chains.items(), key=lambda kv: -snr[kv[0]]):
        fl = match_sources(problem, draws[j], truth[j], windows[j])[:, s]
        for frame in ("eq", "ecl"):
            to = (lambda v: v) if frame == "eq" else to_ecliptic
            vf, vm = to(unit(fl[:, 3], fl[:, 4])), to(unit(mc[:, 3], mc[:, 4]))
            vt = to(unit(truth[j, s, 3], truth[j, s, 4]))
            lt, _ = lon_sinlat(vt)
            (lf, sf), (lm, sm) = lon_sinlat(vf), lon_sinlat(vm)
            lf, lm = wrap(lf, lt), wrap(lm, lt)
            wm, wf = np.array([width(lm), width(sm)]), np.array([width(lf), width(sf)])
            diff = np.array([np.median(lf) - np.median(lm), np.median(sf) - np.median(sm)])
            mf, mm = vf.mean(0), vm.mean(0)
            spread = np.sqrt(np.median(np.arccos(np.clip(vm @ (mm / np.linalg.norm(mm)), -1, 1)) ** 2))
            f0m, f0f = mc[:, 0] * problem.t_obs, fl[:, 0] * problem.t_obs
            design = np.stack([f0m - f0m.mean(), np.ones_like(f0m)], 1)
            slope = np.linalg.lstsq(design, np.stack([lm, sm], 1), rcond=None)[0][0]
            pred = slope * (np.median(f0f) - np.median(f0m)) / wm
            print(f"{j}_{s} {snr[j, s]:7.1f} {frame:>5s} | {diff[0] / wm[0]:7.2f} {diff[1] / wm[1]:7.2f}"
                  f" | {diff[0] / wf[0]:7.2f} {diff[1] / wf[1]:7.2f} | {angle(mf, mm) / spread:8.2f}"
                  f" | {angle(vt, mm) / spread:6.2f} | {pred[0]:6.2f} {pred[1]:6.2f}")
        ve, vte = to_ecliptic(unit(fl[:, 3], fl[:, 4])), to_ecliptic(unit(truth[j, s, 3], truth[j, s, 4]))
        mirror = vte * np.array([1, 1, -1])
        f0_shift = (np.median(fl[:, 0]) - np.median(mc[:, 0])) / width(mc[:, 0])
        print(f"      antipodal side {np.mean(ve @ vte < 0):.3f}, nearer the ecliptic mirror"
              f" {np.mean(ve @ mirror > ve @ vte):.3f}; f0 offset {f0_shift:.2f} MCMC widths;"
              f" ecliptic latitude {np.degrees(np.arcsin(vte[2])):.1f} deg, f0 {truth[j, s, 0] * 1e3:.3f} mHz")

    rows = []
    for j in range(len(truth)):
        fl = match_sources(problem, draws[j], truth[j], windows[j])
        for s in range(truth.shape[1]):
            lf, bf = lon_sinlat(to_ecliptic(unit(fl[:, s, 3], fl[:, s, 4])))
            lt, bt = lon_sinlat(to_ecliptic(unit(truth[j, s, 3], truth[j, s, 4])))
            lf = wrap(lf, lt)
            rows.append([snr[j, s], np.mean(lf < lt), np.mean(bf < bt),
                         (np.median(lf) - lt) / width(lf), (np.median(bf) - bt) / width(bf)])
    rows = np.array(rows)
    print(f"\n{run}: {len(rows)} eval sources, ecliptic; rank of the truth, z = (median - truth) / width")
    for lo, hi in [(0, 15), (15, 100), (100, np.inf)]:
        m = (rows[:, 0] >= lo) & (rows[:, 0] < hi)
        r = rows[m]
        print(f"  SNR {lo}-{hi}: n {m.sum():2d}, rank lambda {r[:, 1].mean():.2f} beta {r[:, 2].mean():.2f},"
              f" mean z {r[:, 3].mean():+.2f} {r[:, 4].mean():+.2f}, rms z"
              f" {np.sqrt((r[:, 3] ** 2).mean()):.2f} {np.sqrt((r[:, 4] ** 2).mean()):.2f}")
