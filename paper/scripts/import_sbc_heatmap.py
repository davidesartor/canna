"""SBC p_min per (f0, SNR) cell for the six reported models (e64.SELECTED), from their validation scorecards.

Every source of the 1000 prior windows of each model's last eval goes to one cell: five f0 ranges, log-uniform over
its band, by the draws' median f0, and five SNR ranges with about the same number of sources each, by the draws'
median SNR (snr_hat). Both are functions of the data alone: stratifying ranks by the true f0 or SNR would condition
on the truth, and the ranks would not be uniform even for the exact posterior. In each cell, the smallest gamma-test
p-value over the eight flow coordinates (canna.lisa.validate.gamma_test), as in the Section 6 summary table.

    cd paper/scripts && JAX_PLATFORMS=cpu nice ../../.venv/bin/python import_sbc_heatmap.py

Writes data/runs/sbc-heatmap.json (local, like the rest of paper/data/).
"""

import glob
import json
import sys
from pathlib import Path

import numpy as np

PAPER = Path(__file__).resolve().parents[1]
REPO = PAPER.parent
sys.path.insert(0, str(REPO / "src"))

from canna.lisa import config  # noqa: E402
from canna.lisa.problem import LisaGB  # noqa: E402
from canna.lisa.samples import load  # noqa: E402
from canna.lisa.validate import gamma_test  # noqa: E402

import e64  # noqa: E402

OUT = PAPER / "data/runs/sbc-heatmap.json"
PROBLEMS = [("narrow", 2), ("narrow", 4), ("narrow", 8), ("full", 4), ("full", 8), ("full", 16)]
COORDS = ["f0", "fdot", "lon", "lat", "re_zp", "im_zp", "re_zm", "im_zm"]
BINS = 5
MIN_SOURCES = 20  # fewer sources in a cell: no p-value

out = {}
for band, n in PROBLEMS:
    run = e64.run_name(band, n)
    card = sorted(glob.glob(str(REPO / f"outputs/runs/{run}/eval/*/scorecard.npz")))[-1]
    s = load(card)
    k = int(s["draws"]) + 1
    # per source (window, slot): the draws' median f0 [Hz] and SNR; row 0 of f0_bins / source_snr is the truth
    f0 = np.median(np.asarray(s["f0_bins"])[:, 1:, :n], axis=1).ravel() / float(s["t_obs"])
    snr = np.asarray(s["snr_hat"])[:, :n].ravel()
    ranks = {c: np.asarray(s[f"rank_{c}"])[:, :n].ravel().astype(int) for c in COORDS}
    lo, hi = LisaGB(**config.load(f"{band}-B").problem).f0_range
    f_edges = np.geomspace(lo, hi, BINS + 1)
    s_edges = np.quantile(snr, np.linspace(0, 1, BINS + 1))
    fi = np.clip(np.searchsorted(f_edges, f0, side="right") - 1, 0, BINS - 1)
    si = np.clip(np.searchsorted(s_edges, snr, side="right") - 1, 0, BINS - 1)
    p_min = np.full((BINS, BINS), np.nan)
    worst = [[None] * BINS for _ in range(BINS)]
    count = np.zeros((BINS, BINS), int)
    for a in range(BINS):
        for b in range(BINS):
            cell = (fi == a) & (si == b)
            count[a, b] = int(cell.sum())
            if count[a, b] < MIN_SOURCES:
                continue
            p = [gamma_test(ranks[c][cell], k)[0] for c in COORDS]
            p_min[a, b] = float(np.min(p))
            worst[a][b] = COORDS[int(np.argmin(p))]
    out[f"{band}-{n}"] = {
        "run": run,
        "eval": Path(card).parent.name,
        "sources": int(f0.size),
        "f_edges_mhz": (f_edges * 1e3).tolist(),
        "snr_edges": s_edges.tolist(),
        # [f0 range][SNR range]
        "p_min": [[None if np.isnan(v) else v for v in row] for row in p_min],
        "worst": worst,
        "count": count.tolist(),
    }
    print(f"{band:6} S={n:2} {run}: {f0.size} sources, SNR edges {np.round(s_edges, 2).tolist()}")
    print("   counts per f0 range:", count.sum(1).tolist())
OUT.write_text(json.dumps(out, indent=1))
print(f"wrote {OUT}")
