# Posterior validation: the tests of `canna.lisa.validate` on a run, and their size and power

`canna.lisa.validate` checks calibration with several kinds of test:

- **SBC ranks** of the truth among the flow draws, per coordinate and per derived quantity, scored
  by the ECDF γ test (α = 0.05).
- **TARP**.
- **Joint pre-ranks**: average rank and band depth.
- **Data-space ranks**: overlap with the data, own SNR and log-likelihood.
- **Proper scores**: energy and per-coordinate CRPS.

Results are split into strata of the draws' median SNR.

This report runs those tests twice:

- on the real injected truth of one run;
- on known errors injected into the draws, to measure what each test can and cannot see.

Only one run is in `data/` so far: `dusty-otter-narrow-XS`. It is a narrow-XS pipeline test
stopped at epoch 4 and evaluated at 32 and 128 ODE steps. The page is
[posterior-validation.html](posterior-validation.html).

## Takeaways

- **The tests hold their size.** On an exact null, the rank tests reject in 2–9% of
  100-injection chunks on the toy, against α = 5%. On the run, the rate is 0–12%, from only 10
  chunks. The proper scores read 0 there by construction (see Method).
- **Per-coordinate SBC is blind to whole classes of error.** On the toy:
  - **Phase or sky from the prior, and dependence shuffled.** Each of these leaves every 1-D
    marginal exact, so the coordinate ranks reject at the null rate.
  - **A 10% prior contamination.** Coordinate ranks reach only about 0.1.
  - **What catches them.** The data-space ranks reject in 97–100% of chunks: overlap,
    log-likelihood, and own SNR for the shuffle. The energy score and the CRPS of the affected
    coordinates also catch them, and joint band depth catches the shuffle (0.94–0.98).
  - **Corollary.** Posterior = prior passes marginal SBC, so a clean rank histogram on its own
    proves nothing.
- **Mild width errors need the right test.**
  - **Narrow ×0.8.** The coordinate ranks catch it in 55–67% of chunks on the toy. Overlap,
    band depth and joint log-likelihood catch it every time.
  - **Wide ×1.25.** The coordinate ranks catch it in 32–48% of chunks. Overlap and
    log-likelihood catch it every time.
  - **On the run.** The strongest tests for width are the amplitude |z±| and own SNR, at 1.0 even
    for ×1.25. Overlap is weaker there (0.28 for ×0.8).
  - **f0 shift.** A 0.5σ shift is caught by the f0 rank (0.99 on the toy, 1.0 on the run). On the
    toy, log-likelihood also catches it (1.0).
- **TARP and average rank add little.**
  - Per-slot TARP mostly repeats the width tests. On the run, joint TARP also catches the z+ phase
    (1.0) and 10% prior draws (0.6–0.8).
  - Average rank is the weakest joint test against everything except the f0 shift.
- **dusty-otter epoch 4 fails every data-space test while nearly passing marginal SBC.** This is
  the blind spot above, on a real flow:
  - **Data-space tests.** Overlap and log-likelihood have p < 0.001 in every slot. The truth fits
    the data better than all 126 draws in almost every injection: mean rank offset +0.38 to
    +0.50, middle-half share 0.00–0.06. Own SNR and |z±| reject as well, with the draws too wide
    on the loud slots (middle-half share 0.60–0.70).
  - **Coordinate ranks.** Most are near calibrated: chirp mass, f0 ± Mc and the z components of the
    quieter slots mostly have p > 0.1. Over 1000 injections, f0 is slightly too wide (middle-half share
    0.55–0.60) and the sky slightly too narrow (0.40–0.46), at p < 0.05.
  - **Joint pre-ranks.** At 32 / 128 steps, joint average rank has p = 0.70 / 0.024 and band depth
    p = 0.057 / 0.043.
  - **Step count.** 32 → 128 ODE steps changes nothing material, for this weak model.
- **TARP on the real truth is invalid in these evals: a key collision, since fixed.**
  - **The bug.** `validate_injection` drew `tarp_ref` from `jr.fold_in(key, 1)`, which in JAX
    equals `key_p` from `jr.split(key, 4)`. The TARP reference point was the truth itself, so
    TARP rejects with offset −0.50 everywhere.
  - **The fix.** The reference now has its own fifth key from `jr.split(key, 5)`. The first four
    keys are unchanged, so later evals draw the same injections. `validation.npz` files written
    before the fix, like dusty-otter's, still carry the bug.
  - **Scope.** The power study is unaffected, because its stand-in truth is a flow draw while the
    reference stays the real truth.

## Method

`test_power.py` builds draws `x [N, M + 1, S, 8]` with the truth in row 0. It runs
`validate.analyse` on them unchanged, then under each perturbation of rows 1..M. A test's rate
is the share of 100-injection chunks in which it rejects at α. For per-slot tests the share is
taken over chunks × slots.

- **Toy (exact posterior).**
  - Each of the 4 slots is an 8-D linear-Gaussian model y = Aθ + σε, with θ ~ N(0, I) and σ
    log-uniform in [0.05, 1] per source.
  - The posterior is Gaussian and drawn exactly: 5000 injections, 126 draws.
  - `f0_bins`, the source SNR, overlap and log-likelihood are the same quantities computed on y,
    so every test runs unchanged.
- **Run (q-as-truth null).**
  - Read from the run's newest `validation.npz` (the scorecard's).
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
- **Real-truth cells.** For each SNR stratum, test, slot and the joint window, the page shows the
  γ-test p-value, the mean rank offset (rank/M − 0.5) and the share of ranks in the middle half.
  That share is 0.5 when calibrated, above 0.5 when the draws are too wide and below when they
  are too narrow.

## How to run

The scorecard writes `validation.npz` first. Then, from the repo root after
`source slurm/<cluster>.env`:

```bash
sbatch $CANNA_SBATCH_GPU reports/diagnostics/posterior-validation/power.sbatch toy
sbatch $CANNA_SBATCH_GPU reports/diagnostics/posterior-validation/power.sbatch <run> LISA_ODE_STEPS=128
```

A run job checks out the run's commit (`slurm/run-src.sh`) and writes
`data/<run>-<ode_steps>.json`. Each job finishes by rebuilding the page with `build_page.py`. To
rebuild the page alone:

```bash
uv run python reports/diagnostics/posterior-validation/build_page.py
```

On Unity, add `"--constraint=sm_75|sm_80|sm_86|sm_89|sm_90"`: the autotuner crashes on sm_52.
