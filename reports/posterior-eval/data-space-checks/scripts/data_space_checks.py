"""Data-space validation tests (overlap, own SNR, log-likelihood ranks) of eval dirs, as data/<setting>.json.

From the repo root, as data_space_checks.sbatch does:
    uv run --frozen python reports/posterior-eval/data-space-checks/scripts/data_space_checks.py \
        --setting <run>-narrow-B-32 --band narrow outputs/runs/<run>/eval/<tag>
"""

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from canna.lisa.samples import load as load_samples
from canna.lisa.validate import COORDS, SNR_BANDS, gamma_test

REPORT = Path(__file__).parents[1]
STRATA = [(0, np.inf), *SNR_BANDS]
HIST_BINS = 16
SCATTER_INJECTIONS = 300
SLOT_TESTS = ["overlap", "own_snr"]
CONTRAST = {
    "abs_zp": "|z+| radius",
    "abs_zm": "|z-| radius",
    "tarp": "TARP",
    "band_depth": "band depth",
}


def stratum_label(lo: float, hi: float) -> str:
    return "all" if lo == 0 and not np.isfinite(hi) else (f">={lo:g}" if not np.isfinite(hi) else f"{lo:g}-{hi:g}")


def p_value(ranks: np.ndarray, k: int) -> float | None:
    """The γ test's p, or None when too few ranks to test."""
    return float(gamma_test(ranks, k)[0]) if len(ranks) >= 10 else None


def rank_panel(ranks: np.ndarray, k: int) -> dict:
    """n, p, ECDF - z with its band, and a coarse rank histogram of one rank set."""
    if len(ranks) < 10:
        return dict(n=len(ranks))
    p, band = gamma_test(ranks, k)
    z = np.arange(1, k) / k
    ecdf = np.array([(ranks < i).mean() for i in range(1, k)]) - z
    hist = np.bincount(np.minimum(ranks * HIST_BINS // k, HIST_BINS - 1), minlength=HIST_BINS)
    return dict(
        n=len(ranks),
        p=float(p),
        ecdf=ecdf.round(3).tolist(),
        band=band.round(3).tolist(),
        hist=hist.tolist(),
    )


def case_record(eval_dir: Path) -> dict:
    """Every number of one eval dir's page: panels per test and stratum, ladder p, raw scatters."""
    scores = d = load_samples(eval_dir / "scorecard.npz")
    raw = {k: d[k] for k in ("source_snr", "overlap", "log_likelihood")}
    k = int(scores["draws"]) + 1
    n, n_slots = scores["snr_hat"].shape
    in_stratum = lambda snr, lo, hi: (snr >= lo) & (snr < hi)

    panels = {}
    for test in SLOT_TESTS:
        r = scores[f"rank_{test}"]
        for slot in [*range(n_slots), "pooled"]:
            panels[f"{test} {slot}"] = []
            for lo, hi in STRATA:
                band = in_stratum(scores["snr_hat"], lo, hi)
                pick = band[:, slot] if slot != "pooled" else band
                ranks = r[:, slot][pick] if slot != "pooled" else r[pick]
                panels[f"{test} {slot}"].append(dict(stratum=stratum_label(lo, hi)) | rank_panel(ranks, k))
    panels["log_likelihood"] = []
    for lo, hi in STRATA:
        pick = in_stratum(scores["joint_snr_hat"], lo, hi)
        panels["log_likelihood"].append(
            dict(stratum=stratum_label(lo, hi)) | rank_panel(scores["joint_rank_log_likelihood"][pick], k)
        )

    # ladder: p over all injections, pooled over slots for per-slot tests
    ladder = {t: p_value(scores[f"rank_{t}"].ravel(), k) for t in SLOT_TESTS}
    ladder["log_likelihood"] = p_value(scores["joint_rank_log_likelihood"], k)
    coord_p = {c: p_value(scores[f"rank_{c}"].ravel(), k) for c in COORDS}
    ladder["coords_min"] = min(coord_p.values())
    ladder |= {t: p_value(scores[f"rank_{t}"].ravel(), k) for t in CONTRAST}
    ladder["joint_tarp"] = p_value(scores["joint_rank_tarp"], k)
    ladder["joint_band_depth"] = p_value(scores["joint_rank_band_depth"], k)

    # raw distributions of the first injections: truth (row 0) against the draws' median
    sub = slice(0, SCATTER_INJECTIONS)
    pair = lambda v: np.stack([v[sub, 0].ravel(), np.median(v[sub, 1:], 1).ravel()], -1)
    scatter = dict(
        overlap=pair(raw["overlap"]),
        own_snr=pair(raw["source_snr"]),
        log_likelihood=pair(raw["log_likelihood"]),
        snr_hat=np.stack([raw["source_snr"][sub, 0].ravel(), scores["snr_hat"][sub].ravel()], -1),
    )
    return dict(
        n=n,
        draws=k - 1,
        ladder=ladder,
        coord_p=coord_p,
        panels=panels,
        scatter={name: np.asarray(v, dtype=np.float64).round(3).tolist() for name, v in scatter.items()},
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", required=True)
    parser.add_argument("--band", required=True, choices=["narrow", "full"])
    parser.add_argument("--smoke", action="store_true", help="the run is a pipeline test")
    parser.add_argument("eval_dirs", nargs="+", type=Path)
    args = parser.parse_args()

    cases = []
    for eval_dir in args.eval_dirs:
        eval_yaml = yaml.safe_load((eval_dir / "eval.yaml").read_text())
        is_mock = eval_yaml.get("weights") == "mock"
        name = eval_dir.name if is_mock else "run"
        print(f"{args.setting}: {name} <- {eval_dir}", flush=True)
        cases.append(
            dict(name=name, mock=is_mock, epoch=eval_yaml.get("epoch"), ode_steps=eval_yaml.get("ode_steps"))
            | case_record(eval_dir)
        )
    out = REPORT / "data" / f"{args.setting}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(
        json.dumps(dict(setting=args.setting, band=args.band, smoke=args.smoke, cases=cases), separators=(",", ":"))
    )
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
