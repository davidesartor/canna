"""Rank calibration of eval dirs: γ-test p, ECDF difference and its band per test, slot and SNR stratum.

Reads each dir's scorecard.npz (validate.summarise's scores beside the samples) and scores it with validate's own γ test and strata.
Run from the repo root, as calibration.sbatch does (no arguments: every mock; a run's eval dirs by argument):
    uv run --frozen python reports/posterior-eval/calibration/scripts/calibration_probe.py [<eval dir> ...]
Each record goes to data/<setting>.json: <run>-<ode_steps> for a run, <band>-<mock> for a mock flow.
"""

import json
import sys
import warnings
from datetime import date
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
from canna.lisa.samples import load as load_samples
from canna.lisa.validate import ALPHA, COORDS, SNR_BANDS, gamma_test

REPORT = Path(__file__).parents[1]
MOCKS = REPORT.parent / "mock-flows" / "data"
DEFAULT_DIRS = [d for band in ("narrow", "full") for d in sorted((MOCKS / band).glob("*"))]
SLOT_TESTS = [*COORDS, "f0_plus_fdot", "f0_minus_fdot", "abs_zp", "abs_zm", "tarp", "average_rank", "band_depth"]
JOINT_TESTS = ["tarp", "average_rank", "band_depth"]
STRIDE_START, STRIDE = 3, 4


def cell(ranks: np.ndarray, k: int, bands: dict) -> dict | None:
    """γ-test p, mean-rank offset, middle-half share and the strided ECDF minus z of one rank set."""
    n = len(ranks)
    if n < 10:
        return None
    p, band = gamma_test(ranks, k)
    bands[str(n)] = band[:, STRIDE_START::STRIDE].round(3).tolist()
    ecdf = np.bincount(ranks, minlength=k).cumsum()[:-1] / n - np.arange(1, k) / k
    return dict(
        n=n,
        p=round(float(p), 4),
        bias=round(float(ranks.mean() / (k - 1) - 0.5), 3),
        middle=round(float(np.mean((ranks >= k / 4) & (ranks < 3 * k / 4))), 3),
        ecdf=ecdf[STRIDE_START::STRIDE].round(3).tolist(),
    )


def read_meta(eval_dir: Path) -> dict:
    """eval.yaml's flat key: value lines."""
    pairs = (line.split(":", 1) for line in (eval_dir / "eval.yaml").read_text().splitlines() if ":" in line)
    return {key.strip(): value.strip() for key, value in pairs}


def setting_of(eval_dir: Path) -> dict:
    """The setting's name, kind and band from the eval dir's path and eval.yaml."""
    meta = read_meta(eval_dir)
    if "mock-flows" in eval_dir.resolve().parts:
        band, mock = eval_dir.parent.name, eval_dir.name
        return dict(setting=f"{band}-{mock}", kind="mock", band=band, mock=mock)
    run = eval_dir.parents[1].name
    band = next((b for b in ("narrow", "full") if f"-{b}-" in run), "unknown")
    return dict(setting=f"{run}-{meta['ode_steps']}", kind="run", band=band, run=run, size=run.rsplit("-", 1)[-1])


def analyse(eval_dir: Path) -> dict:
    """One setting's record."""
    meta = read_meta(eval_dir)
    scores = d = load_samples(eval_dir / "scorecard.npz")
    tarp_is_truth = bool(np.allclose(d["tarp_ref"], d["x"][:, 0]))
    k = int(scores["draws"]) + 1
    n_slots = scores["snr_hat"].shape[1]
    bands: dict = {}
    strata, cells, counts = [], {}, {}
    for lo, hi in [*SNR_BANDS, (0, np.inf)]:
        label = f"{lo:g}-{hi:g}" if np.isfinite(hi) else (f">={lo:g}" if lo else "all")
        slot_mask = (scores["snr_hat"] >= lo) & (scores["snr_hat"] < hi)
        joint_mask = (scores["joint_snr_hat"] >= lo) & (scores["joint_snr_hat"] < hi)
        strata.append(label)
        counts[label] = dict(sources=slot_mask.sum(0).tolist(), injections=int(joint_mask.sum()))
        cells[label] = {}
        for name in SLOT_TESTS:
            r = scores[f"rank_{name}"]
            cells[label][name] = dict(
                slot=[cell(r[slot_mask[:, s], s], k, bands) for s in range(n_slots)],
                pooled=cell(r[slot_mask], k, bands),
                joint=cell(scores[f"joint_rank_{name}"][joint_mask], k, bands) if name in JOINT_TESTS else None,
            )
    return dict(
        **setting_of(eval_dir),
        date=str(date.today()),
        eval_dir=str(eval_dir),
        epoch=int(meta["epoch"]),
        ode_steps=int(meta["ode_steps"]),
        injections=len(scores["joint_snr_hat"]),
        draws=int(scores["draws"]),
        k=k,
        alpha=ALPHA,
        tarp_ref_is_truth=tarp_is_truth,
        stride=[STRIDE_START + 1, STRIDE],
        strata=strata,
        counts=counts,
        tests=SLOT_TESTS,
        joint_tests=JOINT_TESTS,
        bands=bands,
        cells=cells,
    )


def main() -> None:
    dirs = [Path(a) for a in sys.argv[1:]] or DEFAULT_DIRS
    (REPORT / "data").mkdir(exist_ok=True)
    for eval_dir in dirs:
        record = analyse(eval_dir)
        out = REPORT / "data" / f"{record['setting']}.json"
        out.write_text(json.dumps(record, separators=(",", ":")))
        everything = record["cells"]["all"]
        reject = [t for t in SLOT_TESTS if (everything[t]["pooled"] or {"p": 1})["p"] < ALPHA]
        print(f"{record['setting']}: {record['injections']} injections, rejecting (pooled, all SNR): {reject}", flush=True)


if __name__ == "__main__":
    main()
