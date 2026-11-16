"""Summaries of outputs/runs/261103-frame-vs-flow.py (laptop GPU, 2026-11-13; 256 draws per eval window) into
data/runs/frame-vs-flow.json: per model and source group, the medians of frame / MCMC, flow / MCMC and flow / frame f0
widths (bins), and the share of sources whose truth lies within one flow width of the flow's median (a 68% coverage);
for the float32-backbone runs (<cfg>-fp32) the same, to set beside the bf16 ones. numpy only."""

import json
from pathlib import Path

import numpy as np

PAPER = Path(__file__).resolve().parents[1]
DATA = PAPER.parent / "outputs/runs/261103-analysis"
GROUPS = {
    "lt15": lambda s, f: s < 15,
    "15-40": lambda s, f: (s >= 15) & (s < 40),
    "ge40_lo": lambda s, f: (s >= 40) & (f < 1e-2),
    "ge40_hi": lambda s, f: (s >= 40) & (f >= 1e-2),
}

out = {}
for path in sorted(DATA.glob("frame-vs-flow-*.npz")):
    d = np.load(path)
    tag = path.stem.removeprefix("frame-vs-flow-")
    snr, f0 = d["snr"].ravel(), d["f0_hz"].ravel()
    g = lambda k: d[k].ravel()
    rec = {"run": str(d["run"]), "epoch": int(d["epoch"])}
    for name, sel in GROUPS.items():
        m = sel(snr, f0) & np.isfinite(g("mcmc_width"))
        if not m.any():
            continue
        rec[name] = {
            "n": int(m.sum()),
            "frame_over_mcmc": float(np.median(g("frame_sigma")[m] / g("mcmc_width")[m])),
            "flow_over_mcmc": float(np.median(g("flow_width")[m] / g("mcmc_width")[m])),
            "flow_over_frame": float(np.median(g("flow_width")[m] / g("frame_sigma")[m])),
            "flow_within_1": float(np.mean(np.abs(g("flow_offset")[m]) < g("flow_width")[m])),
        }
    out[tag] = rec
    print(tag, {k: (v["flow_over_mcmc"], v["flow_over_frame"]) for k, v in rec.items() if isinstance(v, dict)})
(PAPER / "data/runs/frame-vs-flow.json").write_text(json.dumps(out, indent=1))
