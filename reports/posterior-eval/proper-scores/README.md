# Proper scores: energy score, CRPS and width ratio against the fisher mock (2026-10-30)

## Takeaways

- **Mocks only, no run yet** (src at 8310cb3). Run rows drop in once a run has an eval of the current layout (`crps_fdot` in place of `crps_mc`); the `dusty-otter-narrow-XS` files of 2026-10-24 stay in `data/` but are filtered out of the page.
- **Every eval source is faint** (the mocks ran with `visible_snr = 0`, catalogue windows; draws' median SNR ŝ at most 2.6). The prior mock, the skill floor, then scores about as well as `fisher`, so skill = (floor − score) / (floor − fisher) is undefined (they differ by under 10%) and the page reads the ladder as ratios to `fisher`. Rerun with `visible_snr` on.
- **A raw score has no scale**, so a run is to be read as a skill against the same band's mocks once it exists: 1 is the fisher mock, 0 the prior mock (a data-blind identity flow), below 0 worse than ignoring the data. The comparison is paired when the run's `f`, truth and `tarp_ref` match the mocks' first injections.
- **Mock ladder (ratio to fisher; narrow 200 injections, full 100).**

  | mock | energy | joint | CRPS f0 | read |
  |---|---|---|---|---|
  | wide ×3 | 1.53 (1.49) | 1.49 (1.44) | 1.45 (1.24) | 10 of 11 flag it (width ratio undefined) |
  | narrow ×0.5 | 1.03 (1.03) | 1.05 (1.05) | 1.05 (1.04) | 10 of 11 flag it, lon does not; far less sharp than wide |
  | biased f0 +1σ | 1.04 (1.02) | 1.03 (1.02) | 1.32 (1.20) | f0 CRPS (and 8 of 11 narrow, 7 full) |
  | prior | 0.96 (0.98) | 0.97 (0.99) | 1.11 (1.23) | below fisher almost everywhere (z± CRPS 0.43-0.47 narrow, fdot 0.89) |
  | phases-prior | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | invisible narrow (z± CRPS 0.97-1.01); z± flagged on full |
  | loud-floor | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | equal to fisher: no source reaches its SNR 18.4 cap |

  The energy score is weak against `narrow` and `biased`. At faint SNR the prior is a better posterior than the Fisher mock, so `prior` beats `fisher` on the proper scores and is not a defect to these scores here.
- **Width ratio.** Median flow-over-bound f0 width, narrow band: fisher 15.0, wide 79, narrow 5.3, biased 14.2, prior 1.2. The geometric mean for `wide` collapses to 0 because 57 of its 800 slots have a zero f0 width; read the median.
- **Units caveat.** The scores are in unweighted flow coordinates, so the energy score is dominated by the large-scale coordinates (sky, amplitudes) and barely moves with f0. Read the per-coordinate CRPS next to it.

## What the page shows

[proper-scores.html](proper-scores.html). The top bar picks the setting (a mock ladder, or each run eval) and mean or median; the hash keeps them, e.g. `#setting=mocks-narrow&stat=median`.

1. **How do the runs compare?** The main view. Tiles with the joint energy skill and its paired change, then a table of skill per score (energy per slot, joint, CRPS for the 8 coordinates) and the width ratio per setting, with a paired Δ from the first setting and its bootstrap interval, for a chosen SNR stratum.
2. **Scores by SNR stratum and coordinate.** Skill heatmap for the picked setting (or a picked mock), strata as columns.
3. **Which defects does each score rank correctly?** The mock ladder: each score's ratio to fisher with an interval-excludes-1 marker and the lowest-per-column outline.
4. **What the reference is.** Which mocks, how many injections matched and what was checked.

## Method

`scripts/scores_probe.py` is numpy only on saved files; it reads `scorecard.npz` (`validate.summarise`'s scores, or analyses its validation samples if absent) and the mock ladder from `../mock-flows/data/<band>/<mock>/`.

- **Definitions** (from `validate.py`, in flow coordinates, no normalisation): per-slot energy score = mean distance of the 127 draws to the truth minus half their mean pairwise distance (fair estimator, `m(m−1)` denominator), over the 8 coordinates of one source; joint energy score the same over the 32 coordinates of the window; CRPS the same over one coordinate. Width ratio = robust f0 width (1.4826 MAD) over the monochromatic Fisher bound √3/(π ŝ_own), per slot.
- **Strata** are the draws' median SNR ŝ (a function of y alone, never the true SNR): 0–15, 15–40, 40–100, ≥100; per-slot scores pool the four slots, each by its own ŝ; joint rows use the joint ŝ. The joint score has few injections outside the lowest and highest strata (0 to 8 in the middle two).
- **Skill.** Floor = the prior mock, reference = the fisher mock, both on the same injections and elements. Used only if `f`, truth and `tarp_ref` of the first `min(N, mock N)` injections agree and the draw counts are equal; the checks are recorded in `match`. Otherwise the floor is an in-setting data-blind score (the truth of injection j scored against the draws of j+1, first 200 injections) and skill is 1 − score / floor, with no fisher reference. That floor is not comparable across settings with the prior mock floor (its f0 scale differs), so the page banners it. Skill is null where floor and fisher differ by under 10% of the floor.
- **Not the test-power pitfall.** The mock ladder and run scores use the real truth, never a draw standing in for it, so a proper score is not 0 by construction; the mock ladder is a defect check, not a null.
- **Statistic.** Mean (the proper-score estimator) or median per cell; the width ratio is a geometric mean or median.
- **Intervals.** Paired bootstrap over injections (1000 resamples): the change between two settings of one run as a mean difference, and a mock's ratio to fisher as a ratio of mean scores (geometric mean for the width ratio).

## How to rebuild

CPU only, ~15 s; from the repo root after `source slurm/unity.env`. The first eval dir is the paired baseline; `--mocks` lists bands whose ladder to score. Narrow-B and full-B drop in by naming their eval dirs (`--band` if the run name lacks `-narrow-`/`-full-`):

```bash
sbatch $CANNA_SBATCH_CPU reports/posterior-eval/proper-scores/scripts/scores.sbatch \
    outputs/runs/<run>/eval/<tag-a> outputs/runs/<run>/eval/<tag-b> --mocks narrow full
```

The job writes `data/<run>-<ode_steps>.json` per eval dir and `data/mocks-<band>.json` per band, then rebuilds the page. The page alone (stdlib only):

```bash
uv run --frozen python reports/posterior-eval/proper-scores/scripts/build_page.py
```

## Data

- `data/dusty-otter-narrow-XS-{32,128}.json` (old layout, `crps_mc`, skipped by the page): cells per score and stratum (`mean`, `median`, `mean_all` over all injections, fisher and floor references, skills), `match`, and `paired` against the first setting.
- `data/mocks-{narrow,full}.json`: the same cells per mock plus `ratios[mock][score]` with a bootstrap interval.

Each record has `setting`, `kind` (`run` or `mocks`), `band`, `epoch`, `ode_steps`, `smoke`, `injections` and `draws`.
