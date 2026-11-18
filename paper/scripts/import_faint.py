"""Faint sources against the prior: for every band and source count's selected model, the f0 width (bins) of the flow, of the
joint MCMC reference and of the prior itself (f0-ordered prior draws of the same window), per source below SNR 3, where the
posterior is the prior. Every fourth eval injection, 2048 prior draws each. Writes data/runs/faint-prior.json (local).
Run like import_runs.py: cd <checkout at cc93fb2> && PYTHONPATH=$PWD/src python <repo>/paper/scripts/import_faint.py
"""

import json
from functools import partial
from pathlib import Path

import jax
import jax.random as jr
import numpy as np
import yaml

from canna.lisa.injections import sort_by_f0
from canna.lisa.mcmc import load_reference
from canna.lisa.problem import LisaGB
from canna.lisa.samples import load
from canna.lisa.simulator import sample_physical

import e64  # noqa: E402

PAPER = Path(__file__).resolve().parents[1]
RUNS = PAPER.parent / "outputs/runs"
CONFIGS = [("narrow", 2), ("narrow", 4), ("narrow", 8), ("full", 4), ("full", 8), ("full", 16)]
INVISIBLE = 3.0
runs = {r["run"]: r for r in json.loads((PAPER / "data/runs/source-count.json").read_text())["runs"]}
mad = lambda a: 1.4826 * np.median(np.abs(a - np.median(a, 0)), 0)

out = []
for band, n in CONFIGS:
    run = runs.get(e64.run_name(band, n))  # the e64 model (e64.py); was the lowest eval-catalogue loss
    if not run or not run["evals"]:
        print(f"{band} S={n}: {e64.run_name(band, n)} not scored yet, skipped")
        continue
    config = yaml.safe_load((RUNS / run["run"] / "config.yaml").read_text())
    problem = LisaGB(**config["problem"])
    T = problem.t_obs
    d = load(RUNS / run["run"] / "eval" / run["evals"][-1]["eval"] / "draws.npz")
    ref = e64.reference_for(d["truth"])
    draw = jax.jit(jax.vmap(partial(sample_physical, problem), in_axes=(0, None)))
    flow, mcmc, prior = [], [], []
    for j in range(0, len(d["truth"]), 4):
        m = ref["mcmc"][j]
        faint = d["source_snr"][j] < INVISIBLE
        if not (np.isfinite(m).all() and faint.any()):
            continue
        p = np.asarray(sort_by_f0(draw(jr.split(jr.key(j), 2048), d["f"][j])))
        flow += mad(d["draws"][j][..., 0] * T)[faint].tolist()
        mcmc += mad(m[..., 0] * T)[faint].tolist()
        prior += mad(p[..., 0] * T)[faint].tolist()
    flow, mcmc, prior = map(np.asarray, (flow, mcmc, prior))
    out.append(dict(band=band, n_sources=n, run=run["run"], sources=len(flow),
                    flow_over_prior=float(np.median(flow / prior)), mcmc_over_prior=float(np.median(mcmc / prior)),
                    prior_width_bins=float(np.median(prior))))
    print(out[-1])
(PAPER / "data/runs/faint-prior.json").write_text(json.dumps(dict(invisible_snr=INVISIBLE, configs=out), indent=1))
