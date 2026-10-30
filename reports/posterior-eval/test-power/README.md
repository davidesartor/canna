# Test power: the size and power of `canna.lisa.validate`'s tests on an exact null and known errors (2026-10-30)

## Takeaways

- **The rank tests hold their size.** On an exact null they reject in 2-9% of the 50
  100-injection chunks on the toy, against α = 5%. The proper scores read 0 there by
  construction (see Method). No run row yet: it drops in once a run has an eval of the current
  layout; the 2026-10-24 `dusty-otter-narrow-XS` files stay in `data/` but are not shown.
- **Per-coordinate SBC is blind to whole classes of error.** On the toy:
  - **Phase or sky from the prior, and dependence shuffled.** Each leaves every 1-D marginal
    exact, so the coordinate ranks reject at the null rate.
  - **A 10% prior contamination.** Coordinate ranks reach only about 0.1.
  - **What catches them.** The data-space ranks reject in 97-100% of chunks: overlap,
    log-likelihood, and own SNR for the shuffle. The energy score catches all three (0.95-1.0);
    the CRPS of the affected coordinates catches the phase and sky errors (1.0) and 10% prior
    draws (0.58-0.81), not the shuffle (0.04-0.09). Joint band depth catches the shuffle (0.98).
  - **Corollary.** Posterior = prior passes marginal SBC, so a clean rank histogram on its own
    proves nothing.
- **Mild width errors need the right test.**
  - **Narrow ×0.8.** The coordinate ranks catch it in 55-67% of chunks on the toy. Overlap,
    band depth and joint log-likelihood catch it every time.
  - **Wide ×1.25.** The coordinate ranks catch it in 32-48% of chunks. Overlap and
    log-likelihood catch it every time.
  - **f0 shift.** A 0.5σ shift is caught by the f0 rank (0.99) and by log-likelihood (1.0).
- **TARP and average rank add little.**
  - Per-slot TARP mostly repeats the width tests (0.81 for ×0.8, 0.70 for ×1.25).
  - Average rank is the weakest joint test against everything except the f0 shift.

The same tests on real runs and on the mock flows live in [../calibration/](../calibration/)
(SBC ranks, TARP, joint pre-ranks), [../data-space-checks/](../data-space-checks/) (overlap,
own SNR, log-likelihood) and [../proper-scores/](../proper-scores/) (energy score, CRPS).

## What the page shows

[test-power.html](test-power.html). The top bar picks the setting: the toy, or a run's eval
(none yet).

1. **How often does each test reject, under each error of the draws?** A heatmap of rejection
   rates (%), tests as rows grouped into per-slot ranks, joint ranks over the window and proper
   scores, perturbations as columns, the exact null first. Hover a cell for the test, the
   perturbation and the rate.
2. **What this means**: the takeaways, short.

The setting lives in the URL hash, e.g. `#setting=toy`.

## Method

`scripts/test_power.py` builds draws `x [N, M + 1, S, 8]` with the truth in row 0. It runs
`validate.analyse` on them unchanged, then under each perturbation of rows 1..M. A test's rate
is the share of 100-injection chunks in which it rejects at α. For per-slot tests the share is
taken over chunks × slots.

- **Toy (exact posterior).**
  - Each of the 4 slots is an 8-D linear-Gaussian model y = Aθ + σε, with θ ~ N(0, I) and σ
    log-uniform in [0.05, 1] per source.
  - The posterior is Gaussian and drawn exactly: 5000 injections, 126 draws, 50 chunks.
  - `f0_bins`, the source SNR, overlap and log-likelihood are the same quantities computed on y,
    so every test runs unchanged.
- **Run (q-as-truth null).**
  - Read from the run's newest `scorecard.npz` (its validation samples).
  - Flow draw 1 stands in for the truth and draws 2..127 are the posterior. The draws are then
    exchangeable with the stand-in under the flow itself, so the null holds exactly whatever the
    flow's error.
  - The observations are regenerated from `validate.sample`'s keys, and checked against the stored
    `f` and overlaps, to recompute the data-space quantities of the perturbed draws.
  - 1000 injections give only 10 chunks: steps of 0.1 for joint tests and 0.025 for per-slot tests.
- **Perturbations.** All are of the draws around their own per-injection mean and std:
  - narrow ×0.5 / ×0.8 and wide ×1.25 / ×2;
  - an f0 shift of 0.25σ / 0.5σ;
  - z+ given a uniform random phase;
  - the sky replaced by prior draws;
  - each coordinate permuted independently across draws ("dependence shuffled");
  - 10% / 30% of draws replaced by prior draws.
- **Scores.** A proper score has no null distribution of its own. A chunk counts as a detection
  when the perturbed draws score worse than the unperturbed ones, by a one-sided paired Wilcoxon
  test at α. The null column is therefore 0 by construction: it measures power, not size.

## How to run

From the repo root after `source slurm/<cluster>.env`; the run setting needs the scorecard's
`scorecard.npz` (with validation) first:

```bash
sbatch $CANNA_SBATCH_CPU reports/posterior-eval/test-power/scripts/power.sbatch toy
sbatch $CANNA_SBATCH_GPU --gres=gpu:1 --time=03:00:00 --mem=32G reports/posterior-eval/test-power/scripts/power.sbatch <run> LISA_ODE_STEPS=128
```

A run job checks out the run's commit (`slurm/run-src.sh`) and writes
`data/<run>-<ode_steps>.json`; on Unity add `"--constraint=sm_75|sm_80|sm_86|sm_89|sm_90"`
(the autotuner crashes on sm_52). Each job finishes by rebuilding the page. To rebuild the page
alone (stdlib only):

```bash
uv run --frozen --extra cuda python reports/posterior-eval/test-power/scripts/build_page.py
```

## Data

- `data/toy.json`: the toy, 5000 injections × 126 draws (regenerated 2026-10-30, job 66324869, src at 8310cb3).
- `data/dusty-otter-narrow-XS-{32,128}.json` (old layout, `slot mc`; skipped by the page): the q-as-truth null on `dusty-otter-narrow-XS`, a
  narrow-XS pipeline test stopped at epoch 4, 1000 injections × 126 draws at 32 and 128 ODE
  steps. Its `validation.npz` was regenerated after the TARP key fix (3d2413c), though the fix
  would not touch these numbers anyway: the stand-in truth is a flow draw, not the real truth.

Each record holds `setting`, `injections`, `draws`, `chunk`, `alpha`, `perturbations` and
`power[perturbation][test]`, plus `run`, `epoch` and `ode_steps` for a run.
