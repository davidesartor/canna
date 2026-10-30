# Calibration: SBC ranks, TARP and joint pre-ranks per setting and SNR stratum (2026-10-30)

## Takeaways

- **Mocks only, no run yet.** The page holds the 14 mock settings (src at 8310cb3, jobs 66324725/35/36/37 mocks, 66325248 calibration). Run rows drop in once a run has an eval of the current layout (64-injection grid, `fdot` in place of Mc); the `dusty-otter-narrow-XS` files of 2026-10-24 stay in `data/` but are filtered out.
- **Every eval source is faint.** The mocks ran with `visible_snr = 0` (catalogue windows): the median draw SNR (`snr_hat`) is at most 2.6, so all injections sit in the 0-15 stratum and the 15-40, 40-100 and >=100 strata are empty. Rerun with `visible_snr` on.
- **Clean marginal SBC proves nothing on its own.** The `prior` mock (an identity flow) passes f0, lon, lat, the z± components, TARP, average rank and band depth in the narrow band (its only coordinate rejection is fdot, p = 0.009); in the full band lat (p = 0.016) and Re z+ (0.044) are marginal. The joint log-likelihood rank and the other data-space tests catch it ([../data-space-checks/](../data-space-checks/)); the measured power of each test is in [../test-power/](../test-power/).
- **The ladder behaves as designed** (all slots pooled):
  - `wide` and `narrow` reject f0, fdot, lon, lat, the z± components, the derived ranks, TARP and band depth, both bands.
  - `biased` (f0 +1σ) rejects every test except fdot (p = 0.58 narrow, 0.86 full); f0 mean rank offset -0.19 narrow, -0.14 full.
- **`fisher` is not a clean null at this SNR.** f0, fdot and f0 ± fdot pass, but lon, lat, the z± components, |z±|, TARP and the joint ranks reject (narrow, 200 injections; full, 100): faint sources are non-Gaussian in the flow coordinates. Hence `phases-prior` is not separable from it and `loud-floor` equals it (its width cap starts at source SNR 18.4).

## What the page shows

[calibration.html](calibration.html). The top bar picks the setting (a run, or a mock flow of the narrow or full band) and the SNR stratum of the draws' median SNR (`snr_hat`: 0-15, 15-40, 40-100, >=100, all).

1. **Which tests reject?** Rows are the coordinate ranks (f0, fdot, lon, lat, Re/Im z±), derived ranks (f0 ± fdot, |z±|), TARP and the joint pre-ranks (average rank, band depth); columns are the four slots, all slots pooled and the joint rank over the window. Cells show the γ-test p; hover for n, the mean rank offset and the middle-half share. Click a row to select it.
2. **What does the failing test look like?** The selected test's ECDF-minus-diagonal per slot, pooled and joint, inside the γ test's simultaneous 95% band.
3. **Mock ladder.** γ-test p of the chosen stratum (slots pooled; joint rows over the window), tests against every mock of the band and every run of the band.

The setting, stratum and test live in the URL hash (`#setting=narrow-fisher&stratum=all&test=f0`). A banner on a mock names its defect, and a banner flags an eval whose `tarp_ref` equals the truth (the pre-3d2413c key collision).

## Method

`scripts/calibration_probe.py` reads each eval dir's `scorecard.npz` (`validate.summarise`'s ranks and `snr_hat`, beside the validation samples) and scores it with `canna.lisa.validate`'s own `gamma_test`, `SNR_BANDS` and `ALPHA`, so the p-values match `validate`'s printout and mock-flows. Per stratum it takes the slots with `snr_hat` in the band (per slot), the same slots pooled, and for the joint ranks the injections whose `joint_snr_hat` is in the band. A set under 10 ranks is skipped. The ECDF minus z is saved at every 4th of the 127 grid points, with the band for that n. The p floor is 1/1000 (GAMMA_SIMS), shown as `<.001`.

## How to rebuild

From the repo root after `source slurm/unity.env` (CPU only, never on a login node):

```bash
sbatch $CANNA_SBATCH_CPU reports/posterior-eval/calibration/scripts/calibration.sbatch
sbatch $CANNA_SBATCH_CPU reports/posterior-eval/calibration/scripts/calibration.sbatch outputs/runs/<run>/eval/<tag> [more eval dirs]
```

With no arguments it runs all 14 mocks (~20 s). The real narrow-B and full-B runs drop in by giving their eval dirs (each needs a `scorecard.npz` with validation); a dir under `mock-flows/data/` is a mock, any other is a run, named `<run>-<ode_steps>`. Each job ends by rebuilding the page; the page alone (stdlib only): `uv run --frozen python reports/posterior-eval/calibration/scripts/build_page.py`.

## Data

`data/<setting>.json`, 14 settings on the page: `{narrow,full}-{fisher,wide,narrow,biased,prior,phases-prior,loud-floor}`. Each holds `setting`, `kind` (run or mock), `band`, `epoch`, `ode_steps`, `injections`, `draws`, `k`, `alpha`, `tarp_ref_is_truth`, `strata`, `counts` (sources per slot and injections per stratum), `tests`, `bands` (the γ band per n) and `cells[stratum][test]` = `{slot: [cell x 4], pooled, joint}` with `cell = {n, p, bias, middle, ecdf}`.

## Superseded: reports/diagnostics/posterior-validation/

Its power study is `../test-power/`. Its real-truth cells (γ p, rank offset, middle-half share per stratum, slot and the joint window) are this page, for the run and the mocks. Its overlap, own-SNR and log-likelihood rows belong to `../data-space-checks/` and its energy score and CRPS to `../proper-scores/`. Its run findings (f0 too wide, sky too narrow, step count immaterial) came from the smoke run and are not reproduced until a run eval of the current layout exists. Its TARP key-collision note is stale for the current eval files: `dusty-otter-narrow-XS`'s `validation.npz` now has `tarp_ref` distinct from the truth (re-evaluated 2026-10-24), and TARP reads p = 0.03-0.16 jointly instead of offset -0.50.
