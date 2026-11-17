"""The overlap rank per flow-coordinate group (outputs/analysis-261117/overlap-flow-<cfg>.json, overlap_groups_flow.py on
TREX, 2026-11-18) and the amplitude-rank offsets of the validation pass, for each selected model and its Gaussian mock on
the same 1000 validation injections, into data/runs/overlap-groups.json. Offsets are mean rank / 127 - 0.5 of the truth's
overlap among 127 draws, with the named group of every draw set to the truth's (flow coordinates: frequency = x_f, x_fdot;
sky = l, b; phases = arg z+-; moduli = |z+-|). numpy only."""

import json
from pathlib import Path

import numpy as np

PAPER = Path(__file__).resolve().parents[1]
OUT = PAPER.parent / "outputs"
RUNS = {
    "narrow-2": "261103-n2-narrow-B-e32/eval/40-best-32",
    "narrow-4": "261103-n4-narrow-B/eval/62-best-32",
    "narrow-8": "261103-n8-narrow-B-e32/eval/40-best-32",
    "full-4": "261103-n4-full-B-e32/eval/40-best-32",
    "full-8": "261103-n8-full-B/eval/64-best-32",
    "full-16": "261103-n16-full-B-e32/eval/40-best-32",
}
STRATA = {"snr<7": (0, 7), "snr>=7": (7, np.inf), "all": (0, np.inf)}
RANKS = ("abs_zp", "abs_zm", "own_snr", "overlap")

out = {}
for cfg, run in RUNS.items():
    groups = json.loads((OUT / f"analysis-261117/overlap-flow-{cfg}.json").read_text())["inputs"]
    band = cfg.split("-")[0]
    cards = {
        "flow": OUT / "runs" / run / "scorecard.npz",
        "mock": OUT / f"analysis-261113/mocks/{cfg}/mock-flows/data/{band}/fisher/scorecard.npz",
    }
    rec = {}
    for label, path in cards.items():
        with np.load(path) as s:
            snr_hat, k = s["snr_hat"], 127
            ranks = {
                c: {
                    st: float(s[f"rank_{c}"][(snr_hat >= lo) & (snr_hat < hi)].mean() / k - 0.5)
                    for st, (lo, hi) in STRATA.items()
                }
                for c in RANKS
            }
        rec[label] = dict(
            groups={g: {st: c[st]["offset"] for st in STRATA} for g, c in groups[label]["groups"].items()},
            ranks=ranks,
        )
    out[cfg] = rec
    print(cfg, {lab: {g: round(v["all"], 3) for g, v in r["groups"].items()} for lab, r in rec.items()})

(PAPER / "data/runs/overlap-groups.json").write_text(json.dumps(out, indent=1))
