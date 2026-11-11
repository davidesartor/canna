"""Collect the source-count runs (launched 2026-11-03 at 90735b9) into paper/data/runs/source-count.json.

Reads, per eval of outputs/runs/261103-*: the run's config.yaml and history.npz, the flow/MCMC rows of
261103-vs-mcmc-<date>.json, the truth-relabelled scores of 261103-scores-<date>.json (both written by the
outputs/runs/261103-*.py scripts from each eval's draws.npz), and recomputes from the draws the share of
f0 draws off the joint MCMC's peak. Needs canna.lisa at the runs' code (cc93fb2 = 90735b9 in src/) and
reports/mcmc/data of that checkout, so run it from that checkout with its src on PYTHONPATH:

    cd <checkout at cc93fb2> && PYTHONPATH=$PWD/src python <this repo>/paper/scripts/import_runs.py 2026-11-08

paper/data/ stays local (large-files rule); the paper's numbers_results.py and fig_results-*.py read the JSON.
"""

import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import yaml

from canna.lisa.mcmc import COLUMNS, fold, load_reference
from canna.lisa.physics import FDOT_STD
from canna.lisa.plots import mcmc_data
from canna.lisa.problem import LisaGB
from canna.lisa.samples import load

PAPER = Path(__file__).resolve().parents[1]
# CANNA_RUNS / CANNA_OUT override the paths, to run it on TREX next to the runs
RUNS = Path(os.environ.get("CANNA_RUNS", PAPER.parent / "outputs/runs"))
OUT = Path(os.environ.get("CANNA_OUT", PAPER / "data/runs/source-count.json"))
# 1000-step epoch times of the 30-minute smoke tests on TREX (A100), 2026-11-03, by (band, n_sources)
SMOKE_EPOCH_S = {("narrow", 2): 65, ("narrow", 4): 69, ("narrow", 8): 81, ("full", 4): 72, ("full", 8): 86, ("full", 16): 108}

date = sys.argv[1]
extra = sys.argv[2:] or ["n4-full-main"]  # evals scored separately (the diverged full-4 main run)


def rows_by_eval(prefix: str) -> dict:
    out = {}
    for tag in [date, *extra]:
        path = RUNS / f"261103-{prefix}-{tag}.json"
        if path.is_file():
            out |= {(r["run"], r["eval"]): r for r in json.loads(path.read_text())}
    return out


vs_mcmc, scores = rows_by_eval("vs-mcmc"), rows_by_eval("scores")
# the validation pass of scorecard.npz (261103-validation.py): gamma-test p-values per SNR stratum, by eval
validation = rows_by_eval("validation")
references, reference_runs, records = {}, {}, []
for run_dir in sorted(Path(p) for p in glob.glob(str(RUNS / "261103-n*-B*"))):
    config = yaml.safe_load((run_dir / "config.yaml").read_text())
    band = "narrow" if "narrow" in run_dir.name else "full"
    n = config["problem"]["n_sources"]
    history = np.load(run_dir / "history.npz")
    catalog = history["catalog"][:, :2].sum(1)  # eval-catalog frame + flow NLL per epoch
    fork = config.get("fork_from") or ""
    run = {
        "run": run_dir.name,
        "band": band,
        "n_sources": n,
        "fork_epoch": int(fork.split("/")[1]) if fork else None,
        "commit": config["commit"],
        "f0_range_hz": config["problem"]["f0_range"],
        "response_points": config["problem"]["response_points"],
        "total_steps": config["total_steps"],
        "warmup_steps": config["warmup_steps"],
        "log_interval": config["log_interval"],
        "cooldown_frac": config["cooldown_frac"],
        "epochs_done": int(len(catalog)),
        "epochs_planned": config["total_steps"] // config["log_interval"],
        "smoke_epoch_s_per_1000": SMOKE_EPOCH_S[band, n],
        "catalog": [None if not np.isfinite(c) else float(c) for c in catalog],
        "evals": [],
    }
    for eval_dir in sorted((run_dir / "eval").glob("*")) if (run_dir / "eval").is_dir() else []:
        key = (run_dir.name, eval_dir.name)
        if key not in vs_mcmc or key not in scores:
            print(f"skip {key}: not scored")
            continue
        problem = LisaGB(**config["problem"])
        pid = (str(config["problem"]), config["seed"])
        if pid not in references:
            references[pid] = load_reference(problem, config["seed"])
            reference_runs[pid] = (band, n)
        ref = references[pid]
        draws = load(eval_dir / "draws.npz")
        # f0 draws off the MCMC's peak: farther than max(0.5 bin, 5 MCMC widths) from the MCMC median
        m = ref["mcmc"][..., 0] * problem.t_obs
        f = draws["draws"][..., 0] * problem.t_obs
        median = np.median(m, 1)
        width = 1.4826 * np.median(np.abs(m - median[:, None]), 1)
        tol = np.maximum(0.5, 5 * width)[:, None]
        loud = draws["source_snr"] >= 15
        epoch = int(eval_dir.name.split("-")[0])
        # per source with a reference: where it sits, how close its nearest neighbour is (truth, in bins), the
        # flow's f0 width in bins, and per COLUMNS coordinate the flow/MCMC width ratio
        truth_bins = draws["truth"][..., 0] * problem.t_obs
        sources = []
        for s in mcmc_data(problem, draws, ref)["sources"]:
            j, k = (int(x) for x in s["label"].replace("injection ", "").replace("source ", "").split(", "))
            k -= 1
            others = np.delete(truth_bins[j], k)
            flow_f0 = fold(draws["draws"][j, :, k], draws["truth"][j, k], problem.t_obs)[:, 0]
            flow_f0 = flow_f0[np.isfinite(flow_f0)]
            sources.append({
                "injection": j,
                "slot": k,
                "snr": s["snr"],
                "f0_hz": float(draws["truth"][j, k, 0]),
                "sep_bins": float(np.min(np.abs(others - truth_bins[j, k]))) if len(others) else None,
                "flow_f0_width_bins": float(1.4826 * np.median(np.abs(flow_f0 - np.median(flow_f0)))),
                "ratio": [float(x) for x in s["values"][1]],
            })
        vm = vs_mcmc[key]
        run["evals"].append(
            {
                "eval": eval_dir.name,
                "epoch": epoch,
                "catalog_at_eval": run["catalog"][epoch - 1],
                "mcmc_max_rhat": vm["mcmc_max_rhat"],
                "mcmc_min_ess": vm["mcmc_min_ess"],
                "mcmc_f0_in68": vm["mcmc_f0_in68"],
                "off_peak_snr15": float(np.mean(np.mean(np.abs(f - median[:, None]) > tol, 1)[loud])),
                "mcmc_off_peak_snr15": float(np.mean(np.mean(np.abs(m - median[:, None]) > tol, 1)[loud])),
                "loud_sources": int(loud.sum()),
                "vs_mcmc": vm["rows"],
                "scores": scores[key]["rows"],
                "sources": sources,
                "validation": validation.get(key, {}).get("strata"),
                "validation_injections": validation.get(key, {}).get("injections"),
            }
        )
    records.append(run)
    print(run["run"], [e["eval"] for e in run["evals"]])

# the joint chains alone, per reference: the posterior width of log10 fdot over the prior's conditional width
# at the source's f0 (FDOT_STD), per source, with its SNR
chains = []
for pid, ref in references.items():
    band, n = reference_runs[pid]
    m = np.log10(np.where(ref["mcmc"][..., 1] > 0, ref["mcmc"][..., 1], np.nan))  # [N, D, S]
    width = 1.4826 * np.nanmedian(np.abs(m - np.nanmedian(m, 1, keepdims=True)), 1)  # [N, S]
    ok = np.isfinite(ref["mcmc"]).all(axis=(1, 3))
    chains.append({
        "band": band,
        "n_sources": n,
        "snr": ref["source_snr"][ok].round(3).tolist(),
        "log10_fdot_width_over_prior": (width[ok] / FDOT_STD).round(5).tolist(),
    })

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps({"date": date, "columns": COLUMNS, "runs": records, "chains": chains}, indent=1))
print(f"wrote {OUT}")
