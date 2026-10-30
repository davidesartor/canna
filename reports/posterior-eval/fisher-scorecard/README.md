# Fisher scorecard: is the flow as sharp as the Fisher bound, and are its widths honest? (2026-10-30)

No run is on the page: only the 14 mock settings (src at 8310cb3, job 66325250 on the mock-flows of jobs
66324725/35/36/37). The 2026-10-24 `dusty-otter-narrow-XS-4-average-*` files stay in `data/` (old layout, 10 eval
injections) but are filtered out. Real findings need an eval of a trained run on the current layout.

## Takeaways

- **What the page reads.** Each setting's `scorecard.npz` (the eval's per-source f0 scorecard of
  `src/canna/lisa/scorecard.py`): f0 width over the monochromatic Fisher bound `ideal` = √3/(π·SNR)
  bins, the offset in widths (pulls), in68 / in95 per SNR band, and the rank histogram, for the 64
  eval-grid (8 windows × 8 SNR ranks) and the random injections (200 narrow, 50 full). From `draws.npz`
  it also shows each of the 8 parameters' flow width over the repo's `fisher_samples` width against source SNR.
- **Every source is faint.** The mocks ran with `visible_snr = 0` (catalogue windows): the true source SNR peaks
  at 5.9 (narrow) and 9.0 (full), median 0.03, so only the 0-15 band is populated and only 3-4% of sources are
  "found" (width < 1 bin). Everything below is read at sub-threshold SNR; rerun with `visible_snr` on.
- **Ideal is not the Fisher width at faint SNR.** The narrow fisher mock sits at 0.83× `ideal` (full 1.57×)
  and its f0 flow/Fisher median is 1.79 (full 4.03): the posterior is near the prior, not the Fisher
  Gaussian, so a ratio away from 1.0 is not a defect here.
- **Coverage is optimistic and blind to wide posteriors.** `rank`, `offset`, in68 and in95 use
  `match_sources`, which relabels draws toward the truth. The `prior` mock (an identity flow that
  ignores the data) scores in68 0.78 and in95 0.97 in the narrow band (0.79 / 0.97 full), while its f0 width is
  2.8× the Fisher width (13× full).
- **Mock ladder (narrow band; full in the page).**

  | mock | width/ideal | f0 flow/Fisher | overall in68 | in95 | caught by |
  |---|---|---|---|---|---|
  | fisher | 0.83 | 1.79 | 0.75 | 0.99 | nothing here (the reference) |
  | wide ×3 | 1.93 | 4.72 | 0.97 | 1.00 | width and coverage (too wide) |
  | narrow ×0.5 | 0.49 | 0.99 | 0.50 | 0.76 | coverage; the f0 width ratio stays in 0.8-1.25 |
  | biased +1σ f0 | 0.85 | 1.84 | 0.54 | 0.86 | coverage and pulls only |
  | prior | 1.24 | 2.79 | 0.78 | 0.97 | width; coverage looks nominal |
  | phases-prior | 0.83 | 1.79 | 0.75 | 0.99 | neither: f0 is untouched (psi and phi0 ratios 0.39, 1.31 as in fisher) |
  | loud-floor | 0.83 | 1.79 | 0.75 | 0.99 | nothing: identical to fisher, its cap starts at source SNR 18.4 |

  Full band: fisher 1.57 / 4.03 / 0.80, wide 3.65 / 11.5 / 0.96, narrow 0.81 / 1.78 / 0.59, biased 1.63 / 4.20 / 0.61, prior 4.88 / 13.1 / 0.79.
- **Fisher card on the fisher mock.** Median flow/Fisher over all sources (narrow): f0 1.79, fdot 0.34, A 12.2,
  lon 0.47, lat 0.91, psi 0.39, iota 0.31, phi0 1.32. The mock is Gaussian on the same Fisher plus prior, so
  ratios away from 1 are the prior capping widths (A, f0) or the yardstick, not flow error; the cause was not investigated.

## What the page shows

[fisher-scorecard.html](fisher-scorecard.html). The top bar picks the setting: a run's eval, or a mock
(band × mock). Links lead to the sibling pages.

1. **How wide is f0 against the Fisher bound?** f0 width / `ideal` against SNR, eval and random
   injections apart, median per SNR band, lines at 1.0 and the fisher mock's median.
2. **Are the offsets consistent with the widths?** Pulls against SNR, and their histogram against N(0, 1).
3. **Is the truth inside the credible intervals?** in68 and in95 per SNR band (all, eval or random) with
   Wilson whiskers, and the rank histogram with its ±2σ band.
4. **How wide is each parameter against the Fisher forecast?** Where `draws.npz` exists: per source,
   flow width / Fisher width for the chosen parameter against SNR, and a table of the medians for all 8.
5. **Does the scorecard see each mock's defect?** A table per band with the width and coverage views and
   whether each flags (f0 flow/Fisher outside 0.8 to 1.25; overall in68 outside 0.58 to 0.78).
6. **Caveats**, with the numbers above computed live from the mock data.

The setting, subset and parameter are kept in the URL hash. A banner sits on top of each setting.

## Method

`scripts/fisher_probe.py` (numpy only, no model) reads each eval dir:

- `scorecard.npz` rows go to `data/<setting>.json` unchanged (SNR, width, ideal, offset, rank, in68,
  in95, eval flag, injection), plus a per-SNR-band table (all, eval, random). Bands are 0 to 15,
  15 to 40, 40 to 100, ≥ 100, as in `scorecard.py`; "found" is width < 1 bin.
- `draws.npz`, if present: width = 1.4826 × MAD along the draw axis, for the `matched` draws and the
  `fisher` draws, per injection, source and parameter. Longitude, psi and phi0 are wrapped around the
  truth (period 2π) before the MAD. The ratio is flow over Fisher.

Caveats built into the quantities:

- `ideal` is the monochromatic bound, not the fisher mock's width (0.83× it in narrow at faint SNR).
- `rank`, `offset` and the coverage flags go through `match_sources` (a permutation in f0, log A and
  sky), which pulls draws toward the truth.
- `fisher` is `fisher_samples`, centred on the truth: a width yardstick, not a calibrated reference.
- Four sources share a window, and the 64 eval injections are 8 windows × 8 SNR ranks, a stratified set, so intervals treat
  sources as independent.

## How to rebuild

From the repo root, after `source slurm/unity.env` (CPU, a minute):

```bash
# the defaults: every mock-flows setting
sbatch $CANNA_SBATCH_CPU reports/posterior-eval/fisher-scorecard/scripts/fisher.sbatch
# a real run (narrow-B, full-B) once its eval dir exists; any number of dirs
sbatch $CANNA_SBATCH_CPU reports/posterior-eval/fisher-scorecard/scripts/fisher.sbatch outputs/runs/<run>/eval/<tag>
# the page alone (stdlib, fine on a login node)
uv run --frozen python reports/posterior-eval/fisher-scorecard/scripts/build_page.py
```

The sbatch ends with `build_page.py`. With dirs given it only writes those settings; existing
`data/*.json` stay.

## Data

`data/<setting>.json`, one per setting: `mock-<band>-<mock>` (14 on the page) and `<run>-<tag>` (the two old-layout dusty-otter files are skipped).
Each holds the metadata, `rows` (column arrays of the scorecard), `bands`, and, with draws, `fisher`
(`snr`, `ratio`, `flow_width`, `fisher_width` as [injection][source][parameter], and `median_ratio`).
