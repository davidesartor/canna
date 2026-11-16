"""Label-free f0 widths of long MCMC traces (long_chain.py / long_chain_prior.py) against the prior, the packed
reference and the flow. Post-processing only: reads traces, runs no chain.

Why: the traces are f0-sorted per draw. Once faint sources roam the window they cross the loud ones, so a loud
source's f0 rank mixes with its neighbours and its rank width blows up (compare_prior_start.py read 70-316x the
reference). Here a loud source (SNR >= 7) takes, per draw, the slot nearest its true f0; a faint one (SNR < 3, where
the posterior is the prior) keeps its rank and is compared with the f0-ordered prior width of the same window, as in
paper/scripts/import_faint.py. Widths are 1.4826 MAD in frequency bins, per quarter of the chain (both/all seeds
and walkers pooled) to show drift, plus per seed over the second half. "found" is the share of draws with any slot
within FOUND_BINS of the source's true f0 (the posterior weight of the source being localised), with the prior's
share as the chance baseline. fold() already returns f0 in bins; the raw prior draws are in Hz.

Usage (from ~/canna-oneshot, LISA_CONFIG + LISA_PROBLEM set):
  PYTHONPATH=src python label_free_widths.py <out.json> <trace dir> <run> <eval>
"""

import argparse
import json
from functools import partial
from pathlib import Path

import jax
import jax.random as jr
import numpy as np

from canna.lisa import config as run_config
from canna.lisa.injections import sort_by_f0
from canna.lisa.mcmc import fold, load_reference
from canna.lisa.problem import LisaGB
from canna.lisa.samples import load
from canna.lisa.simulator import sample_physical

parser = argparse.ArgumentParser()
parser.add_argument("out")
parser.add_argument("traces", type=Path)
parser.add_argument("run")
parser.add_argument("eval")
parser.add_argument("--runs", type=Path, default=Path("outputs/runs"))
args = parser.parse_args()

config = run_config.load()
problem = LisaGB(**config.problem)
T = problem.t_obs
ref = load_reference(problem, config.seed)
draws = load(args.runs / args.run / "eval" / args.eval / "draws.npz")
prior_draw = jax.jit(jax.vmap(partial(sample_physical, problem), in_axes=(0, None)))
mad = lambda a: 1.4826 * np.median(np.abs(a - np.median(a, 0)), 0)
FAINT, LOUD = 3.0, 7.0
FOUND_BINS = 0.5


def widths(off_f0, truth_f0, snr):
    """off_f0 [N, S] rank offsets in Hz -> per-source width in bins: rank MAD (faint), nearest-slot MAD (loud)."""
    w = mad(off_f0)
    absolute = off_f0 + truth_f0
    for s in np.flatnonzero(snr >= LOUD):
        near = np.take_along_axis(absolute, np.abs(absolute - truth_f0[s]).argmin(1)[:, None], 1)[:, 0]
        w[s] = mad(near - truth_f0[s])
    return w


def found(absolute, truth_f0):
    """absolute [N, S] f0 in bins -> per source, the share of draws with some slot within FOUND_BINS of its truth."""
    return (np.abs(absolute[:, :, None] - truth_f0[None, None, :]).min(1) < FOUND_BINS).mean(0)


records = []
for j in sorted({int(p.stem.removeprefix("inj").split("_")[0]) for p in args.traces.glob("inj*_seed*.npz")}):
    files = sorted(args.traces.glob(f"inj{j}_seed*.npz"))
    seeds = [np.load(f)["trace"][..., 0] for f in files]  # [W, T, S] f0 offsets from the f0-sorted truth
    t0 = np.load(files[0])
    truth_f0, snr = t0["truth_folded"][:, 0], np.asarray(t0["snr"])
    n_t = seeds[0].shape[1]
    quarters, found_q = [], []
    for q in range(4):
        part = np.concatenate([x[:, q * n_t // 4:(q + 1) * n_t // 4].reshape(-1, x.shape[2]) for x in seeds])
        quarters.append(widths(part, truth_f0, snr))
        found_q.append(found(part + truth_f0, truth_f0))
    per_seed = [widths(x[:, n_t // 2:].reshape(-1, x.shape[2]), truth_f0, snr) for x in seeds]

    truth = draws["truth"][j]
    ft = np.stack([fold(truth[s][None], truth[s], T)[0] for s in range(len(truth))])
    rel = lambda a: np.stack([fold(a[:, s], truth[s], T) for s in range(len(truth))], 1)[..., 0] - ft[:, 0]
    w_ref = widths(rel(ref["mcmc"][j]), ft[:, 0], snr)
    w_flow = widths(rel(draws["draws"][j]), ft[:, 0], snr)
    p = np.asarray(sort_by_f0(prior_draw(jr.split(jr.key(j), 4096), draws["f"][j])))
    w_prior = mad(p[..., 0] * T)
    f_ref = found(rel(ref["mcmc"][j]) + ft[:, 0], ft[:, 0])
    f_flow = found(rel(draws["draws"][j]) + ft[:, 0], ft[:, 0])
    f_prior = found(p[..., 0] * T, ft[:, 0])
    records.append(dict(injection=j, seeds=len(files), snr=snr.tolist(),
                        chain_quarters=[q.tolist() for q in quarters], chain_seeds_late=[s.tolist() for s in per_seed],
                        reference=w_ref.tolist(), flow=w_flow.tolist(), prior=w_prior.tolist(),
                        found_chain_quarters=[q.tolist() for q in found_q], found_reference=f_ref.tolist(),
                        found_flow=f_flow.tolist(), found_prior=f_prior.tolist()))
    late = np.median(np.stack([quarters[2], quarters[3]]), 0)
    for name, sel in (("SNR<3", snr < FAINT), ("3-7", (snr >= FAINT) & (snr < LOUD)), ("SNR>=7", snr >= LOUD)):
        if sel.any():
            m = lambda a: float(np.median(a[sel]))
            print(f"inj {j:3d} {name:6} n {sel.sum():2d} | chain/prior by quarter "
                  + " ".join(f"{m(q / w_prior):5.2f}" for q in quarters)
                  + f" | ref/prior {m(w_ref / w_prior):5.2f} flow/prior {m(w_flow / w_prior):5.2f}"
                  + f" | chain(late)/ref {m(quarters[3] / w_ref):6.2f} flow/chain(late) {m(w_flow / quarters[3]):5.2f}"
                  + f" | seed1/seed0 {m(per_seed[-1] / per_seed[0]):5.2f}"
                  + " | found chain Q1-4 " + " ".join(f"{m(q):4.2f}" for q in found_q)
                  + f" ref {m(f_ref):4.2f} flow {m(f_flow):4.2f} prior {m(f_prior):4.2f}", flush=True)

Path(args.out).write_text(json.dumps(dict(run=args.run, eval=args.eval, traces=str(args.traces), faint_snr=FAINT,
                                          loud_snr=LOUD, found_bins=FOUND_BINS, injections=records), indent=1))
print(f"wrote {args.out}")
