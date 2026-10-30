# Reference comparison: the flow's draws against the MCMC reference (2026-10-24)

Every number for the run is **smoke**: `dusty-otter-narrow-XS` is a 2,000-step narrow-XS pipeline test stopped at
epoch 4. It shows that the comparison works end to end, not how good the method is. Real findings need narrow-B and
full-B (training on trex, ETA 2026-10-25); they drop in by re-running the sbatch under [How to rebuild](#how-to-rebuild).

## Takeaways

- **The comparison reads known widths correctly.** The mocks of [mock-flows](../mock-flows/) go through the same code
  against the same chains (10 injections, 40 sources per-source, 16 jointly). Median flow/MCMC width over the eight
  linear parameters, per-source reference:
  - `fisher` 0.91-1.03 (joint reference 0.69-1.01, so the linearised Gaussian is a little wide in log A, psi, sin i, phi0);
  - `wide` 2.7-3.1 (3x built in), `narrow` 0.46-0.52 (0.5x built in); `biased` has the `fisher` widths (it only shifts f0);
  - `prior` is 4950x in f0 and 13-36x elsewhere (log Mc 1.0x); `phases-prior` matches `fisher` except psi 12x, phi0 24x;
  - `loud-floor` is 3-8x in most parameters and 20x pinned phi+-.
- **The smoke run is far too wide, close to the prior in f0.** Per-source (40 sources) the median flow/MCMC width is
  837x in f0 (53-8450x), 22.5x in phi0, 10x in log A and sin i, 8x in RA and Dec, 5.5x in psi, and 0.99x in log Mc
  (the posterior is the prior there). Against the joint MCMC (16 sources) the medians are 873x f0, 5.4x phi0, 4.4x log A.
- **The gap grows with SNR.** The median phi+ width over the narrower MCMC phi+- width is 17x below SNR 15, 51x at 15-100,
  144x at 100-300 and 294x above 300 (8, 12, 9, 11 sources); the true posterior narrows with SNR and the flow does not.
- **Too wide, not offset.** The truth's rank among the flow's draws falls in the central 68% for 78% of source-parameter
  pairs and in the central 95% for 97%, both above nominal, as an over-wide posterior gives.
- **Mind the sampler.** The draws come from 32 uniform x-pred ODE steps, which collapse narrow posteriors late in the
  path ([sampler-convergence](../sampler-convergence/)). A collapse only narrows, so a flow this much wider than the
  reference is the network's doing; once a trained run is within a few times the reference width, evaluate it with more
  steps before reading its widths.

## What the page shows

[`reference-comparison.html`](reference-comparison.html) is standalone (8 MB, all data inlined) and works from
`file://`, light or dark. The top-bar setting picker chooses the run or a mock; links to the sibling reports are under
the title. The sidebar sets the reference (per-source or joint MCMC), the parameter (pinned φ± or any of the ten) and
the selected source; a dot clicked in any card selects that source everywhere. State is in the URL hash. A smoke
banner appears for the run, a mock banner for the mocks, and a sampler banner for runs with an ODE sampler.

1. **Which parameters is the flow too wide in?** Flow width over reference width per source, log axis, with row medians.
2. **Does the gap grow with SNR?** The selected parameter's ratio against source SNR (φ₀ as rings on pinned φ±).
3. **Does the truth fall where the flow says?** Rank histogram of the truth among the draws, and 68%/95% coverage per
   parameter. Needs no MCMC, so it covers every band.
4. **What does the posterior look like next to the joint MCMC?** Offset scatter of any two parameters for a window and
   source (flow blue, MCMC gray, Fisher Gaussian orange), with a table of widths, W1, seed agreement, coverage of the
   MCMC median and the fraction of draws more than a bin off in f₀.

"What this means" is written from the selected setting's numbers.

## Method

`scripts/reference_probe.py` takes `name=eval_dir` pairs and writes `data/<name>.json` for each.

- **Injection check.** Before comparing, the probe asserts each eval dir's `truth`, `f` and `source_snr` are identical to
  `eval_injections(problem, seed)` of the base config (max |Δtruth| = 0, exact), so the draws are for the injections
  the chains were run on. All seven mocks pass; the mocks use the narrow-B config (seed 0), whose problem hash
  (`ad8a87a575`) is the chains' and the run's. Noise keys are not saved in `draws.npz`: they follow from the seed and
  the injection index, as in the repo's eval and as `mock_eval.py` replays them, so the joint chains' noise realisation is
  the one every setting was drawn on.
- **Per-source widths.** Each source's draws are matched to the truth and folded (`mcmc.fold`); the parameters are
  f₀ in bins, log ḟ, log A, sky, ψ, sin ι, φ₀, plus φ± = φ₀ ± 2ψ read off `physical_to_flow` at the chart's phase
  reference. Widths are 1.4826·MAD, circular (√(−2 ln R)) for φ±, against each source's marginal of the joint MCMC
  (`mcmc.load_chains`, `outputs/mcmc/joint-<id>`, up to 40,000 draws per injection over its seeds, matched to the
  truth). "Pinned" is the narrower of the MCMC's φ₊ and φ₋. The truth's rank among the draws is recorded too (φ± cut
  opposite the circular mean).
- **Joint comparison.** Four injections (a mid-SNR one in every other window of the band, `--windows` to change)
  against the same chains, kept per seed (`outputs/mcmc/joint-<id>/inj<j>_seed<k>.npz`): widths, W1 over the MCMC width, the Fisher Gaussian's width,
  seed agreement, coverage of the MCMC median in the flow's 68%/95%, and 250 draws per set for the page. The
  likelihood's symmetry under the folds is checked at the truth for each window.
- **The data predates the joint-only reference.** `data/*.json` were written against the earlier per-source chains
  (others at their truth) and windows 2, 5, 7, 9, which the takeaways quote; a rebuild replaces both.
- **No MCMC is regenerated.** The probe only reads the cached chains. The full band has none, so a full-B setting has
  card 3 only; the full-band mocks are therefore not settings.
- `scripts/record.py` holds the numpy helpers for the joint window record, `scripts/build_page.py` (stdlib only) writes
  the page from `data/*.json`.

## How to rebuild

From the repo root, after `source slurm/unity.env` (CPU only, ~4 min for four settings; the sbatch ends by rebuilding
the page). The settings of this report:

```bash
sbatch $CANNA_SBATCH_CPU reports/posterior-eval/reference-comparison/scripts/reference.sbatch mock-fisher=reports/posterior-eval/mock-flows/data/narrow/fisher mock-wide=reports/posterior-eval/mock-flows/data/narrow/wide mock-narrow=reports/posterior-eval/mock-flows/data/narrow/narrow mock-biased=reports/posterior-eval/mock-flows/data/narrow/biased
sbatch $CANNA_SBATCH_CPU reports/posterior-eval/reference-comparison/scripts/reference.sbatch mock-prior=reports/posterior-eval/mock-flows/data/narrow/prior mock-phases-prior=reports/posterior-eval/mock-flows/data/narrow/phases-prior mock-loud-floor=reports/posterior-eval/mock-flows/data/narrow/loud-floor
```

The real narrow-B run drops in by pointing at its eval dir (it needs `draws.npz`, as written by `canna.lisa.eval`, and
the narrow per-source and joint chains in `outputs/mcmc`; full-B settings have no chains and give card 3 only):

```bash
sbatch $CANNA_SBATCH_CPU reports/posterior-eval/reference-comparison/scripts/reference.sbatch --run <narrow-B run> <run>-32=outputs/runs/<run>/eval/<epoch>-average-32
sbatch $CANNA_SBATCH_CPU reports/posterior-eval/reference-comparison/scripts/reference.sbatch --run <narrow-B run> <run>-128=outputs/runs/<run>/eval/<epoch>-average-128
```

The page alone: `uv run --frozen python reports/posterior-eval/reference-comparison/scripts/build_page.py`.

## Data

`data/<setting>.json`, ~1 MB each: setting metadata (run, config, steps, eval tag, epoch, ODE steps), `widths` (per
eval source: SNR, sin ι, flow widths, truth ranks, MCMC widths) and `joint` (per window and source: comparison rows,
coverage, 250 draws per set). Settings: `dusty-otter-narrow-XS-32` and the seven `mock-*` (narrow band).

`dusty-otter-narrow-XS-32` was **converted, not recomputed**: `outputs/runs/dusty-otter-narrow-XS/eval/4-average-32/`
no longer has a `draws.npz`, so the record is the earlier flow-vs-reference widths and joint comparison, passed through
`record.py` (same page format, 250 instead of 400 draws kept). Its `note` field says so. A `4-average-128` setting needs
that eval's draws, which are also missing; re-running `canna.lisa.eval` for the run would restore both. Re-running the
probe on a run needs the eval dir's draws to exist.

The velocity probe ([`velocity_probe.py`](../../architecture/data-dependence/scripts/velocity_probe.py), the "network reads the data's phase" finding) is not here; it lives in the
[data-dependence](../../architecture/data-dependence/) report.
