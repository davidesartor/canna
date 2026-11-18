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
import e64  # noqa: E402

# the e64 models (e64.py); a config is kept only once its overlap-flow json was computed on that model's scorecard
RUNS = {}
for _band, _n in e64.CONFIGS:
    _ev = e64.eval_dir(_band, _n)
    if _ev is not None:
        RUNS[f"{_band}-{_n}"] = f"{e64.run_name(_band, _n)}/eval/{_ev.name}"
STRATA = {"snr<7": (0, 7), "snr>=7": (7, np.inf), "all": (0, np.inf)}
RANKS = ("abs_zp", "abs_zm", "own_snr", "overlap")

out = {}
for cfg, run in RUNS.items():
    # e64 recomputations land as overlap-flow-<cfg>-e64.json; the 2026-11-18 ones hold the earlier picks
    paths = [OUT / f"analysis-261117/overlap-flow-{cfg}{s}.json" for s in ("-e64", "")]
    rec_runs = [(p, json.loads(p.read_text())) for p in paths if p.is_file()]
    rec_runs = [(p, r) for p, r in rec_runs if r["run"] == run.split("/")[0]]
    if not rec_runs:
        print(f"{cfg}: no overlap-flow json for {run}, skipped")
        continue
    groups = rec_runs[0][1]["inputs"]
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
