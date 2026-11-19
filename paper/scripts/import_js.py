"""Jensen-Shannon divergence of the flow's marginals from the joint MCMC's, as eq. 30 of Delmond et al. (2026).

Per source with a reference chain, per COLUMNS coordinate (f0 in bins, ln fdot, ln A, equatorial lon/lat, psi,
sin iota, phi0; psi/phi0 folded and lon wrapped onto the truth, canna.lisa.mcmc.fold), for the model of each band
and source count with the lowest eval-catalogue loss (numbers_results.selected). Densities are histograms on B
bins at the quantiles of the pooled flow + MCMC draws; the bins then depend on ranks only, so the estimator's
floor for two equal distributions is distribution-free, and it is simulated here for the same draw counts. Base-2
logarithms, so 0 <= JS <= 1.

    cd paper/scripts && JAX_PLATFORMS=cpu nice ../../.venv/bin/python import_js.py

Reads outputs/runs/261103-*/eval/<eval>/draws.npz and reports/mcmc/data/eval-*.npz (matched by the truth);
writes data/runs/js-divergence.json (local, like the rest of paper/data/).
"""

import glob
import json
import sys
from pathlib import Path

import numpy as np
import yaml

PAPER = Path(__file__).resolve().parents[1]
REPO = PAPER.parent
sys.path.insert(0, str(REPO / "src"))

from canna.lisa.mcmc import COLUMNS, fold  # noqa: E402
from canna.lisa.problem import LisaGB  # noqa: E402
from canna.lisa.samples import load  # noqa: E402

RUNS = REPO / "outputs/runs"
OUT = PAPER / "data/runs/js-divergence.json"
BINS = 32
NULL_TRIALS = 2000
CONFIGS = [("narrow", 2), ("narrow", 4), ("narrow", 8), ("full", 4), ("full", 8), ("full", 16)]


def js_divergence(p: np.ndarray, q: np.ndarray, bins: int = BINS) -> float:
    """Eq. 30 in bits between the samples p and q, on `bins` equal-mass bins of the pooled sample."""
    pooled = np.concatenate([p, q])
    edges = np.quantile(pooled, np.linspace(0, 1, bins + 1)[1:-1])
    # side="right" puts ties (e.g. a coordinate pinned at a box edge) in one bin, the same for both samples
    hp = np.bincount(np.searchsorted(edges, p, side="right"), minlength=bins) / len(p)
    hq = np.bincount(np.searchsorted(edges, q, side="right"), minlength=bins) / len(q)
    m = 0.5 * (hp + hq)
    kl = lambda a: np.sum(a[a > 0] * np.log2(a[a > 0] / m[a > 0]))
    return float(0.5 * kl(hp) + 0.5 * kl(hq))


def null_floor(n_p: int, n_q: int, rng: np.random.Generator) -> np.ndarray:
    """The estimator on two independent samples of one continuous distribution (rank-based: any one will do)."""
    return np.array([js_divergence(rng.standard_normal(n_p), rng.standard_normal(n_q)) for _ in range(NULL_TRIALS)])


# 2026-11-19: both outputs now read the reported models (e64.SELECTED). Was: `import_js.py main`: the 64-epoch main runs instead of
# the lowest-loss model; full S=4 then needs its Student-t continuation (MAIN_RUN), skipped until it lands
MAIN = len(sys.argv) > 1 and sys.argv[1] == "main"
import e64  # noqa: E402  (2026-11-19: the reported models, e64.SELECTED)
if MAIN:
    OUT = PAPER / "data/runs/js-divergence-main.json"


def selected_eval(band: str, n: int) -> tuple[Path, dict] | None:
    """The eval with the lowest eval-catalogue loss among the main run and its cooldown forks (or the main run)."""
    if True:  # 2026-11-19: both outputs read the reported model (e64.SELECTED); was: MAIN only, else lowest loss
        name = e64.run_name(band, n)
        evals = sorted((RUNS / name / "eval").glob("*/draws.npz")) if (RUNS / name / "eval").is_dir() else []
        if not evals:
            print(f"{band} S={n}: no eval of {name} yet, skipped")
            return None
        return evals[-1].parent, {"run": name}
    runs = json.loads((PAPER / "data/runs/source-count.json").read_text())["runs"]
    family = [r for r in runs if r["band"] == band and r["n_sources"] == n and r["evals"]]
    best = min(family, key=lambda r: r["evals"][-1]["catalog_at_eval"])
    return RUNS / best["run"] / "eval" / best["evals"][-1]["eval"], best


references = {}
for path in sorted(glob.glob(str(REPO / "reports/mcmc/data/eval-*.npz"))):
    ref = load(path)
    references[path] = ref

rng = np.random.default_rng(0)
out = {"bins": BINS, "columns": COLUMNS, "models": []}
for band, n in CONFIGS:
    picked = selected_eval(band, n)
    if picked is None:
        continue
    eval_dir, run = picked
    config = yaml.safe_load((eval_dir.parents[1] / "config.yaml").read_text())
    t_obs = LisaGB(**config["problem"]).t_obs
    d = load(eval_dir / "draws.npz")
    matches = [p for p, r in references.items() if r["truth"].shape == d["truth"].shape and np.allclose(r["truth"], d["truth"])]
    assert len(matches) == 1, (eval_dir, matches)
    ref = references[matches[0]]
    sources = []
    for j, s in zip(*np.nonzero(np.isfinite(ref["mcmc"]).all(axis=(1, 3)))):
        truth = d["truth"][j, s]
        m = fold(ref["mcmc"][j, :, s], truth, t_obs)
        f = fold(d["draws"][j, :, s], truth, t_obs)
        f = f[np.isfinite(f).all(axis=1)]
        sources.append({
            "injection": int(j),
            "slot": int(s),
            "snr": float(d["source_snr"][j, s]),
            "mcmc_ess": float(ref["ess"][j, s]),
            "n_flow": int(len(f)),
            "n_mcmc": int(len(m)),
            "js": [js_divergence(f[:, c], m[:, c]) for c in range(len(COLUMNS))],
        })
    n_f, n_m = sources[0]["n_flow"], sources[0]["n_mcmc"]
    floor = null_floor(n_f, n_m, rng)
    out["models"].append({
        "band": band,
        "n_sources": n,
        "run": run["run"],
        "eval": eval_dir.name,
        "reference": Path(matches[0]).name,
        "null": {"n_flow": n_f, "n_mcmc": n_m, "median": float(np.median(floor)),
                 "p95": float(np.quantile(floor, 0.95)), "p99": float(np.quantile(floor, 0.99))},
        "sources": sources,
    })
    js = np.array([x["js"] for x in sources])
    print(f"{band:6s} S={n:2d} {run['run']}/{eval_dir.name}: {len(sources)} sources, floor median {np.median(floor):.4f}"
          f" p95 {np.quantile(floor, 0.95):.4f}; median JS per column", np.round(np.median(js, 0), 3))

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(out))
print(f"wrote {OUT}")
