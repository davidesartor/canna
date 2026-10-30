"""Proper scores of eval dirs and mock ladders as skill against the band's fisher mock; numpy only.

Each eval dir (outputs/runs/<run>/eval/<tag>) or --mocks band gives one data/<setting>.json. The first eval dir
named is the baseline of the paired comparison. From the repo root, as scores.sbatch does:
    uv run --frozen python reports/posterior-eval/proper-scores/scripts/scores_probe.py \
        outputs/runs/<run>/eval/<tag-a> outputs/runs/<run>/eval/<tag-b> --mocks narrow full
"""

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from canna.lisa.samples import load as load_samples
from canna.lisa.validate import COORDS, SNR_BANDS, analyse, energy_score

REPORT = Path(__file__).parents[1]
MOCK_DATA = REPORT.parent / "mock-flows" / "data"
MOCKS = ["fisher", "wide", "narrow", "biased", "prior", "phases-prior", "loud-floor"]
SCORES = ["energy", "joint_energy"] + [f"crps_{c}" for c in COORDS] + ["width_ratio"]
STRATA = ["all"] + [
    f"{lo:g}-{hi:g}" if np.isfinite(hi) else f">={lo:g}" for lo, hi in SNR_BANDS
]
SMOKE_STEPS = 20_000
SHUFFLE_INJECTIONS = 200
BOOTSTRAP = 1000
MIN_SPAN = 0.1


def load(path: Path) -> dict:
    """scorecard.npz's validation keys the check needs, and its scores (or analysed on the fly)."""
    d = load_samples(path / "scorecard.npz")
    samples = {k: d[k] for k in ("f", "tarp_ref", "epoch", "ode_steps")} | {
        "truth": d["x"][:, 0]
    }
    scores = d if "draws" in d else analyse(d)
    return samples | {"scores": scores, "path": str(path)}


def score_arrays(scores: dict) -> dict:
    """Every score as [N, S] (per slot) or [N] (joint), with the stratifying snr of the same shape."""
    out = {
        "energy": (scores["energy"], scores["snr_hat"]),
        "joint_energy": (scores["joint_energy"], scores["joint_snr_hat"]),
    }
    for c, name in enumerate(COORDS):
        out[f"crps_{name}"] = (scores["crps"][:, :, c], scores["snr_hat"])
    out["width_ratio"] = (scores["width_ratio"], scores["snr_hat"])
    return out


def strata_masks(snr: np.ndarray) -> dict:
    """Boolean mask per stratum over the draws' median snr (a function of y alone)."""
    masks = {"all": np.ones(snr.shape, bool)}
    for label, (lo, hi) in zip(STRATA[1:], SNR_BANDS):
        masks[label] = (snr >= lo) & (snr < hi)
    return masks


def stat(a: np.ndarray, mask: np.ndarray) -> tuple[int, float, float]:
    """Count, mean and median of a over the mask; nan when empty."""
    v = a[mask]
    return (
        (int(v.size), float(v.mean()), float(np.median(v)))
        if v.size
        else (0, np.nan, np.nan)
    )


def log_scale(name: str, a: np.ndarray) -> np.ndarray:
    """The width ratio on a log scale (its mean is geometric); every other score as is."""
    return np.log(a) if name == "width_ratio" else a


def clean(x):
    """nan to None for json."""
    return None if x is None or not np.isfinite(x) else float(x)


def per_injection_sums(
    a: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per-injection sum of a and count over the mask, [N] each."""
    a2, m2 = a.reshape(len(a), -1), mask.reshape(len(a), -1)
    return (a2 * m2).sum(1), m2.sum(1).astype(float)


def bootstrap(
    num: np.ndarray,
    den: np.ndarray,
    rng: np.random.Generator,
    ratio_of: np.ndarray | None = None,
) -> tuple[float, float]:
    """Percentile 95% interval over injections of sum(num)/sum(den) with injections resampled."""
    idx = rng.integers(0, len(num), (BOOTSTRAP, len(num)))
    d = den[idx].sum(1)
    stat_ = num[idx].sum(1) / np.where(d == 0, np.nan, d)
    lo, hi = np.nanpercentile(stat_, [2.5, 97.5])
    return float(lo), float(hi)


def skill(own: float, fisher: float, floor: float) -> float | None:
    """(floor - own) / (floor - fisher): 1 at the fisher mock's score, 0 at the data-blind floor; None when they coincide."""
    if not np.isfinite(floor - fisher) or abs(floor - fisher) < MIN_SPAN * abs(floor):
        return None
    return clean((floor - own) / (floor - fisher))


def flow_cells(arrays: dict, n_use: int, refs: dict | None, floor: dict) -> dict:
    """Per score and stratum: own mean/median on the first n_use injections, the references on the same elements, skills."""
    out = {}
    for name, (a, snr) in arrays.items():
        masks = strata_masks(snr)
        out[name] = {}
        for label, mask in masks.items():
            sub = mask.copy()
            sub[n_use:] = False
            n, mean, median = stat(log_scale(name, a), sub)
            mean, median = (np.exp(mean), np.exp(median)) if name == "width_ratio" else (mean, median)
            cell = {
                "n": n,
                "mean": clean(mean),
                "median": clean(median),
                "mean_all": clean(np.exp(stat(log_scale(name, a), mask)[1]) if name == "width_ratio" else stat(a, mask)[1]),
            }
            if name != "width_ratio" and n:
                f_mean = f_median = None
                if refs is not None:
                    _, f_mean, f_median = stat(refs["fisher"][name], sub[: len(refs["fisher"][name])])
                    _, p_mean, p_median = stat(refs["prior"][name], sub[: len(refs["prior"][name])])
                else:
                    _, p_mean, p_median = stat(floor[name], sub[: len(floor[name])])
                cell |= {"floor_mean": clean(p_mean), "floor_median": clean(p_median)}
                if refs is not None:
                    cell |= {
                        "fisher_mean": clean(f_mean),
                        "fisher_median": clean(f_median),
                        "skill_mean": skill(mean, f_mean, p_mean),
                        "skill_median": skill(median, f_median, p_median),
                    }
                else:
                    cell |= {
                        "skill_mean": clean(1 - mean / p_mean),
                        "skill_median": clean(1 - median / p_median),
                    }
            elif name == "width_ratio" and refs is not None and n:
                cell["fisher_median"] = clean(stat(refs["fisher"][name], sub[: len(refs["fisher"][name])])[2])
            out[name][label] = cell
    return out


def shuffled_floor(path: Path, n_use: int) -> dict:
    """In-setting data-blind floor: injection j's truth scored against injection j+1's draws (first n_use injections)."""
    x = load_samples(path / "scorecard.npz")["x"][:n_use]
    truth, draws = x[:, :1], np.roll(x[:, 1:], -1, axis=0)
    mixed = np.concat([truth, draws], 1)
    by_slot = np.moveaxis(mixed, 1, 2)
    floor = {"energy": np.stack([energy_score(by_slot[j]) for j in range(n_use)])}
    floor["joint_energy"] = np.array([energy_score(mixed[j].reshape(1, mixed.shape[1], -1))[0] for j in range(n_use)])
    for c, name in enumerate(COORDS):
        floor[f"crps_{name}"] = np.stack([energy_score(by_slot[j][..., c : c + 1]) for j in range(n_use)])
    return floor


def band_of(run: str, override: str | None) -> str:
    """narrow or full from the run name (config name <band>-<size>), or the override."""
    if override:
        return override
    for band in ("narrow", "full"):
        if f"-{band}-" in run:
            return band
    raise SystemExit(f"cannot tell the band of {run}; pass --band")


def run_setting(path: Path, band: str, rng: np.random.Generator) -> dict:
    """One record for an eval dir: its cells against the band's fisher mock, and what was matched."""
    run, tag = path.parents[1].name, path.name
    eval_cfg = yaml.safe_load((path / "eval.yaml").read_text())
    run_cfg = yaml.safe_load((path.parents[1] / "config.yaml").read_text())
    steps = run_cfg.get("total_steps")
    own = load(path)
    arrays = score_arrays(own["scores"])
    n = len(own["f"])
    fisher, prior = load(MOCK_DATA / band / "fisher"), load(MOCK_DATA / band / "prior")
    n_use = min(n, len(fisher["f"]))
    checks = {
        "f": bool(
            np.allclose(own["f"][:n_use], fisher["f"][:n_use], rtol=1e-6, atol=1e-9)
        ),
        "truth": bool(
            np.allclose(
                own["truth"][:n_use], fisher["truth"][:n_use], rtol=1e-5, atol=1e-6
            )
        ),
        "tarp_ref": bool(
            np.allclose(
                own["tarp_ref"][:n_use],
                fisher["tarp_ref"][:n_use],
                rtol=1e-5,
                atol=1e-6,
            )
        ),
        "draws": bool(own["scores"]["draws"] == fisher["scores"]["draws"]),
    }
    matched = all(checks.values())
    print(
        f"{path}: {n} injections, mock has {len(fisher['f'])}, matched {matched} {checks}",
        flush=True,
    )
    if matched:
        refs = {
            "fisher": {k: v[0] for k, v in score_arrays(fisher["scores"]).items()},
            "prior": {k: v[0] for k, v in score_arrays(prior["scores"]).items()},
        }
        floor, reference = (
            None,
            "the fisher and prior mocks of the same band, on the same injections",
        )
    else:
        n_use = min(n, SHUFFLE_INJECTIONS)
        refs, floor = None, shuffled_floor(path, n_use)
        reference = "an in-setting data-blind floor: the truth scored against another injection's draws"
    return {
        "setting": f"{run}-{eval_cfg['ode_steps']}",
        "kind": "run",
        "run": run,
        "tag": tag,
        "band": band,
        "epoch": int(own["epoch"]),
        "ode_steps": int(eval_cfg["ode_steps"]),
        "total_steps": steps,
        "smoke": steps is not None and steps < SMOKE_STEPS,
        "injections": n,
        "draws": int(own["scores"]["draws"]),
        "match": {
            "n_used": n_use,
            "mock_injections": len(fisher["f"]),
            "checks": checks,
            "matched": matched,
            "reference": reference,
        },
        "flows": {"run": flow_cells(arrays, n_use, refs, floor)},
        "_arrays": arrays,
        "_f": own["f"],
    }


def paired(record: dict, base: dict, rng: np.random.Generator) -> dict | None:
    """Per score and stratum: mean of (record - base) over the same elements, relative to base, with a bootstrap interval."""
    if len(record["_f"]) != len(base["_f"]) or not np.allclose(
        record["_f"], base["_f"]
    ):
        return None
    out = {}
    for name, (a, snr) in record["_arrays"].items():
        b = base["_arrays"][name][0]
        masks = strata_masks(base["_arrays"][name][1])
        out[name] = {}
        for label, mask in masks.items():
            if not mask.any():
                continue
            diff_sum, count = per_injection_sums(log_scale(name, a) - log_scale(name, b), mask)
            base_sum, _ = per_injection_sums(log_scale(name, b) if name == "width_ratio" else b, mask)
            mean_diff = diff_sum.sum() / count.sum()
            lo, hi = bootstrap(diff_sum, count, rng)
            out[name][label] = {
                "diff": clean(mean_diff),
                "lo": clean(lo),
                "hi": clean(hi),
                "rel": clean(np.expm1(mean_diff) if name == "width_ratio" else diff_sum.sum() / base_sum.sum()),
            }
    return {"baseline": base["setting"], "scores": out}


def mock_setting(band: str, rng: np.random.Generator) -> dict:
    """One record for a band's mock ladder: every mock's cells and its paired ratio to fisher with an interval."""
    loaded = {m: load(MOCK_DATA / band / m) for m in MOCKS}
    arrays = {m: score_arrays(v["scores"]) for m, v in loaded.items()}
    refs = {
        "fisher": {k: v[0] for k, v in arrays["fisher"].items()},
        "prior": {k: v[0] for k, v in arrays["prior"].items()},
    }
    n = len(loaded["fisher"]["f"])
    flows, ratios = {}, {}
    for m in MOCKS:
        flows[m] = flow_cells(arrays[m], n, refs, {})
        ratios[m] = {}
        for name, (a, snr) in arrays[m].items():
            f = refs["fisher"][name]
            mask = np.ones(a.shape, bool)
            if name == "width_ratio":
                num, cnt = per_injection_sums(np.log(a / f), mask)
                lo, hi = bootstrap(num, cnt, rng)
                ratios[m][name] = {
                    "ratio": clean(np.exp(num.sum() / cnt.sum())),
                    "lo": clean(np.exp(lo)),
                    "hi": clean(np.exp(hi)),
                }
            else:
                num, cnt = per_injection_sums(a, mask)
                den, _ = per_injection_sums(f, mask)
                lo, hi = bootstrap(num, den, rng)
                ratios[m][name] = {
                    "ratio": clean(num.sum() / den.sum()),
                    "lo": clean(lo),
                    "hi": clean(hi),
                }
    return {
        "setting": f"mocks-{band}",
        "kind": "mocks",
        "band": band,
        "epoch": -1,
        "ode_steps": 0,
        "smoke": False,
        "injections": n,
        "draws": int(loaded["fisher"]["scores"]["draws"]),
        "flows": flows,
        "ratios": ratios,
    }


def save(record: dict) -> None:
    """Write data/<setting>.json without the private arrays."""
    out = REPORT / "data" / f"{record['setting']}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(
        json.dumps({k: v for k, v in record.items() if not k.startswith("_")}, indent=1)
    )
    print(f"saved {out}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "evals",
        nargs="*",
        type=Path,
        help="eval dirs; the first is the paired baseline",
    )
    parser.add_argument("--band", help="narrow or full; default from the run name")
    parser.add_argument(
        "--mocks", nargs="*", default=[], help="bands whose mock ladder to score"
    )
    args = parser.parse_args()
    rng = np.random.default_rng(0)
    records = [
        run_setting(p, band_of(p.parents[1].name, args.band), rng) for p in args.evals
    ]
    for record in records:
        record["paired"] = (
            paired(record, records[0], rng) if record is not records[0] else None
        )
        save(record)
    for band in args.mocks:
        save(mock_setting(band, rng))


if __name__ == "__main__":
    main()
