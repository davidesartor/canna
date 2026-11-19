"""SBC of the reported model of every band and source count (e64.SELECTED; the Section 4 summary table).

Per model, from the validation scorecard of its last eval (1000 prior windows, ranks of the truth among 127
draws): per flow coordinate the gamma-test p-value (canna.lisa.validate.gamma_test) and the share of truths in
the central 68% and 95%, and the share of draws the truth beats in joint log-likelihood. The same for the joint
MCMC from its packed eval reference (128 windows). Full S=4 reads its Student-t continuation (MAIN_RUN) and is
skipped until that eval lands.

    cd paper/scripts && JAX_PLATFORMS=cpu nice ../../.venv/bin/python import_sbc.py

Writes data/runs/sbc-main.json (local, like the rest of paper/data/).
"""

import glob
import json
import sys
from pathlib import Path

import numpy as np

PAPER = Path(__file__).resolve().parents[1]
REPO = PAPER.parent
sys.path.insert(0, str(REPO / "src"))

from canna.lisa.samples import load  # noqa: E402
from canna.lisa.validate import gamma_test  # noqa: E402

OUT = PAPER / "data/runs/sbc-main.json"
# packed MCMC reference per problem (reports/mcmc/data/eval-<problem id>.npz)
PID = {("narrow", 2): "a9a1998079", ("narrow", 4): "cc4b9e9aa0", ("narrow", 8): "ad9cb6a159",
       ("full", 4): "13653a71f0", ("full", 8): "2ca7ffaadb", ("full", 16): "6a6e86da4a"}
import e64  # noqa: E402  (2026-11-19: the reported models, e64.SELECTED)
COORDS = ["f0", "fdot", "lon", "lat", "re_zp", "im_zp", "re_zm", "im_zm"]


def sbc(s: dict) -> dict:
    k = int(s["draws"]) + 1
    cs = []
    for c in COORDS:
        r = np.asarray(s[f"rank_{c}"]).ravel().astype(int)
        u = (r + 0.5) / k
        cs.append((gamma_test(r, k)[0], np.mean((u > 0.16) & (u < 0.84)), np.mean((u > 0.025) & (u < 0.975))))
    i = int(np.argmin([c[0] for c in cs]))
    ll = np.asarray(s["joint_rank_log_likelihood"]).ravel().astype(int)
    return {
        "windows": int(len(ll)),
        "in68": float(np.mean([c[1] for c in cs])),
        "in95": float(np.mean([c[2] for c in cs])),
        "p": {c: float(x[0]) for c, x in zip(COORDS, cs)},
        "min_p": float(cs[i][0]),
        "worst": COORDS[i],
        "loglik_rank": float(ll.mean() / (k - 1)),
    }


out = {}
for (band, n), pid in PID.items():
    run = e64.run_name(band, n)
    cards = sorted(glob.glob(str(REPO / f"outputs/runs/{run}/eval/*/scorecard.npz")))
    if not cards:
        print(f"{band} S={n}: no eval of {run} yet, skipped")
        continue
    flow = sbc(load(cards[-1]))
    mcmc = sbc(dict(np.load(REPO / f"reports/mcmc/data/eval-{pid}.npz", allow_pickle=True)))
    out[f"{band}-{n}"] = {"run": run, "eval": Path(cards[-1]).parent.name, "flow": flow, "mcmc": mcmc}
    print(f"{band:6} S={n:2} {run}: flow 68% {flow['in68']:.3f} min p {flow['min_p']:.2g} ({flow['worst']}); "
          f"mcmc 68% {mcmc['in68']:.3f} min p {mcmc['min_p']:.2g} ({mcmc['worst']})")
OUT.write_text(json.dumps(out, indent=1))
print(f"wrote {OUT}")
