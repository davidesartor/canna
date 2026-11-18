"""Draws for the corner figures (fig_corners.py): per band and source count, the selected model's eval (numbers_results.selected)
on one crowded window, its closest pair of loud sources with similar SNR, flow, joint MCMC and Fisher draws.

Pair rule: neighbours in f0 with both SNR >= 15 and an SNR ratio below 2, the closest in f0; if none, the closest pair with
both SNR >= 7. Draws are relabelled to the injections by f0 order and folded in (psi, phi0) (canna.lisa.mcmc.fold), stored
as offsets from the folded truth in mcmc.COLUMNS, beside the figure in data/corners/<band>-<n>.npz (paper/data stays local).
Run from a checkout of the runs' code holding reports/mcmc/data, with its src on PYTHONPATH:

    cd <checkout at cc93fb2> && PYTHONPATH=$PWD/src python <this repo>/paper/scripts/import_corners.py
"""

import json
from pathlib import Path

import numpy as np
import yaml

from canna.lisa.mcmc import COLUMNS, fold, load_reference
from canna.lisa.problem import LisaGB
from canna.lisa.samples import load

PAPER = Path(__file__).resolve().parents[1]
RUNS = PAPER.parent / "outputs/runs"
OUT = PAPER / "data/corners"
import e64  # noqa: E402

CONFIGS = [("narrow", 2), ("narrow", 4), ("narrow", 8), ("full", 4), ("full", 8), ("full", 16)]

runs = {r["run"]: r for r in json.loads((PAPER / "data/runs/source-count.json").read_text())["runs"]}


def selected(band: str, n: int) -> dict | None:
    """The e64 model (e64.py); was the lowest eval-catalogue loss among the main run and its forks."""
    r = runs.get(e64.run_name(band, n))
    return r if r and r["evals"] else None


def pick(truth_bins, snr, ok):
    """(injection, slot) of the pair's lower-f0 source: the closest qualifying neighbours."""
    best = None
    for loud in (15, 7):
        for j in range(len(snr)):
            for k in range(snr.shape[1] - 1):
                a, b = snr[j, k], snr[j, k + 1]
                if not (ok[j, k] and ok[j, k + 1] and min(a, b) >= loud):
                    continue
                if loud == 15 and max(a, b) / min(a, b) >= 2:
                    continue
                sep = abs(truth_bins[j, k + 1] - truth_bins[j, k])
                if best is None or sep < best[0]:
                    best = (sep, j, k)
        if best:
            return best
    raise SystemExit("no pair")


OUT.mkdir(parents=True, exist_ok=True)
for band, n in CONFIGS:
    run = selected(band, n)
    if run is None:
        print(f"{band} S={n}: not scored yet, skipped")
        continue
    ev = run["evals"][-1]["eval"]
    config = yaml.safe_load((RUNS / run["run"] / "config.yaml").read_text())
    problem = LisaGB(**config["problem"])
    d = load(RUNS / run["run"] / "eval" / ev / "draws.npz")
    ref = e64.reference_for(d["truth"])
    T = problem.t_obs
    ok = np.isfinite(ref["mcmc"]).all(axis=(1, 3))
    sep, j, k = pick(d["truth"][..., 0] * T, d["source_snr"], ok)
    sets = {}
    for name, arr in (("flow", d["draws"][j]), ("mcmc", ref["mcmc"][j]), ("fisher", d["fisher"][j])):
        cols = []
        for s in (k, k + 1):
            truth = d["truth"][j, s]
            x = fold(arr[:, s], truth, T) - fold(truth[None], truth, T)[0]
            cols.append(x)
        x = np.concatenate(cols, axis=1)  # [N, 16]: source k's 8 columns, then source k+1's
        sets[name] = x[np.isfinite(x).all(axis=1)].astype(np.float32)
    path = OUT / f"{band}-{n}.npz"
    np.savez(path, **sets, columns=np.array(COLUMNS), run=run["run"], eval=ev, injection=j, slots=np.array([k, k + 1]),
             snr=d["source_snr"][j, [k, k + 1]], sep_bins=sep, window_snr=d["snr"][j], f_hz=d["f"][j])
    print(f"{band}-{n}: {run['run']} {ev}, injection {j}, slots {k},{k + 1}, SNR {np.round(d['source_snr'][j, [k, k + 1]], 1)},"
          f" separation {sep:.2f} bins -> {path.name}")
