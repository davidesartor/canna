"""Summaries of the robustness probe and the sampler sweep (reports/posterior-eval/{robustness,sampler-convergence} at
cc93fb2; laptop GPU, 2026-11-13; outputs/runs/261103-analysis/{robustness,sampler}-<cfg>.json) into data/runs/probes.json.

Sampler: per step count, the median over sources (SNR >= 15, and all) of each source's f0 width over its width at the most
steps run, and the median |offset| / width. Robustness: per case, the perturbed sources' and the others' median f0 width
over the same source's baseline width, the median |offset| / width in f0 (a confident wrong answer: large, with the width
not grown), and the share found. numpy only."""

import json
from pathlib import Path

import numpy as np

PAPER = Path(__file__).resolve().parents[1]
DATA = PAPER.parent / "outputs/runs/261103-analysis"
out = {"sampler": {}, "robustness": {}}

for path in sorted(DATA.glob("sampler-*.json")):
    r = json.loads(path.read_text())
    cfg = path.stem.removeprefix("sampler-")
    snr = np.array([s["snr"] for s in r["sources"]])
    cells = sorted(r["cells"] + [r["eval_cell"] | {"steps": r["eval_ode_steps"]}], key=lambda c: c["steps"])
    ref = np.array(cells[-1]["width"])[:, 0]
    rows = []
    for c in cells:
        w, off = np.array(c["width"])[:, 0], np.abs(np.array(c["offset"])[:, 0])
        rows.append(dict(steps=c["steps"], seconds=c.get("seconds"), rule=c.get("rule", "eval"),
                         width_over_max_loud=float(np.median((w / ref)[snr >= 15])) if (snr >= 15).any() else None,
                         width_over_max_all=float(np.median(w / ref)),
                         offset_over_width_loud=float(np.median((off / w)[snr >= 15])) if (snr >= 15).any() else None))
    out["sampler"][cfg] = dict(run=r["run"], max_steps=cells[-1]["steps"], eval_steps=r["eval_ode_steps"], rows=rows)
    print(cfg, [(x["steps"], round(x["width_over_max_loud"] or np.nan, 3)) for x in rows])

for path in sorted(DATA.glob("robustness-*.json")):
    r = json.loads(path.read_text())
    cfg = path.stem.removeprefix("robustness-")
    base = {(s["injection"], s["base_slot"]): s for s in r["sources"] if s["case"] == "baseline"}
    cases = {}
    for case in dict.fromkeys(s["case"] for s in r["sources"]):
        rows = [s for s in r["sources"] if s["case"] == case and s["width"][0]]
        rec = {}
        for name, sel in (("faint", lambda s: s["snr"] < 7), ("loud", lambda s: s["snr"] >= 7)):
            pick = [s for s in rows if sel(s) and (s["injection"], s["base_slot"]) in base
                    and base[s["injection"], s["base_slot"]]["width"][0]]
            if not pick:
                continue
            b = [base[s["injection"], s["base_slot"]] for s in pick]
            rec[name] = dict(
                n=len(pick),
                width_over_baseline=float(np.median([s["width"][0] / q["width"][0] for s, q in zip(pick, b)])),
                offset_over_width=float(np.median([abs(s["offset"][0]) / s["width"][0] for s in pick])),
                baseline_offset_over_width=float(np.median([abs(q["offset"][0]) / q["width"][0] for q in b])),
                within_one=float(np.mean([abs(s["offset"][0]) < s["width"][0] for s in pick])),
                found=float(np.mean([s["found"] for s in pick])),
            )
        slots = [s for s in r["slots"] if s["case"] == case]
        rec["claimed_slot_snr"] = float(np.median([s["snr"] for s in slots])) if slots else None
        cases[case] = rec
    for case in {s["case"] for s in r["slots"]} - set(cases):  # cases with no matched source (empty window)
        cases[case] = {"claimed_slot_snr": float(np.median([s["snr"] for s in r["slots"] if s["case"] == case]))}
    out["robustness"][cfg] = dict(run=r.get("run"), injections=r.get("injections"), draws=r.get("draws"), cases=cases)
    for case, rec in cases.items():
        print(cfg, f"{case:22s}", {k: {kk: round(vv, 2) for kk, vv in v.items()} if isinstance(v, dict) else v for k, v in rec.items()})

(PAPER / "data/runs/probes.json").write_text(json.dumps(out, indent=1))
