"""Rank ECDFs of the validation pass (tab:results-validation) for the rank-ECDF figure: for every band and source
count's selected model (lowest eval-catalogue loss, as import_faint.py) and its Gaussian mock on the same injections,
the ECDF of the truth's rank minus uniform, ECDF(z) - z at z = i/k, for the joint log-likelihood ranks (one per
window) and the overlap ranks (one per source), with the simultaneous 95% band of the γ test (Säilynoja+22,
canna.lisa.validate.gamma_test). Reads the runs' scorecard.npz (outputs/runs/<run>/eval/<eval>/) and the mocks'
(outputs/analysis-261113/mocks/<band>-<n>/, TREX job 17657536, 2026-11-13). Writes data/runs/ranks.npz (local).
Run like import_faint.py: PYTHONPATH=<checkout at cc93fb2>/src python import_ranks.py
"""

import json
from pathlib import Path

import numpy as np

from canna.lisa.samples import load
from canna.lisa.validate import gamma_test

PAPER = Path(__file__).resolve().parents[1]
OUT = PAPER.parent / "outputs"
CONFIGS = [("narrow", 2), ("narrow", 4), ("narrow", 8), ("full", 4), ("full", 8), ("full", 16)]
QUANTITIES = {"loglik": "joint_rank_log_likelihood", "overlap": "rank_overlap"}
runs = {r["run"]: r for r in json.loads((PAPER / "data/runs/source-count.json").read_text())["runs"]}


def ecdf(r: np.ndarray, k: int) -> tuple[np.ndarray, float, np.ndarray]:
    """Ranks on 0..k-1 -> ECDF(z) - z at z = i/k (i = 1..k-1), the γ p-value, the band [2, k-1]."""
    z = np.arange(1, k) / k
    p, band = gamma_test(r, k)
    return np.mean(r[:, None] < np.arange(1, k), 0) - z, float(p), band


import e64  # noqa: E402

save = {}
for band, n in CONFIGS:
    run = runs.get(e64.run_name(band, n))  # the e64 model (e64.py); was the lowest eval-catalogue loss
    if not run or not run["evals"]:
        print(f"{band} S={n}: {e64.run_name(band, n)} not scored yet, skipped")
        continue
    flow = load(OUT / "runs" / run["run"] / "eval" / run["evals"][-1]["eval"] / "scorecard.npz")
    mock = load(OUT / f"analysis-261113/mocks/{band}-{n}/mock-flows/data/{band}/fisher/scorecard.npz")
    for name, s in (("flow", flow), ("mock", mock)):
        k = int(s["draws"]) + 1
        for q, key in QUANTITIES.items():
            r = np.asarray(s[key]).ravel().astype(int)
            d, p, b = ecdf(r, k)
            tag = f"{band}-{n}/{name}/{q}"
            save[f"{tag}/ecdf"], save[f"{tag}/band"] = d, b
            save[f"{tag}/k"], save[f"{tag}/p"], save[f"{tag}/n"] = k, p, len(r)
            print(f"{tag:28} n {len(r):6d} k {k} p {p:.3g} offset {r.mean() / (k - 1) - 0.5:+.3f}")
    save[f"{band}-{n}/run"] = run["run"]
np.savez(PAPER / "data/runs/ranks.npz", **save)
