# Data-space checks: do re-simulated draws reproduce the data? (2026-10-30)

## Takeaways

- **Mocks only, no run yet** (src at 8310cb3). Run rows drop in once a run has an eval of the current layout; the `dusty-otter-narrow-XS` files of 2026-10-24 stay in `data/` but are filtered out of the page.
- **Every eval source is faint**: the mocks ran with `visible_snr = 0` (catalogue windows), `snr_hat` is at most 2.6, so all injections are in the 0-15 stratum and the other strata are empty. Rerun with `visible_snr` on.
- **These tests catch what marginal SBC misses.** The `prior` mock (identity flow) passes own SNR (p = 0.76 narrow, 0.57 full), TARP, average rank and band depth, and is rejected by the joint log-likelihood rank (p = 0.001 narrow, < 0.001 full) and by the overlap rank (p = 0.003, 0.007).
- **Width and shift defects** (`wide`, `narrow`, `biased`) reject own SNR and the joint log-likelihood rank in both bands (p < 0.001), but overlap passes `wide` (p = 0.09 narrow, 0.10 full) and `biased` (0.21, 0.15), and `narrow` on the full band (0.18).
- **`fisher` is not a null for these tests.** It rejects own SNR and the joint log-likelihood rank in both bands (p < 0.001); overlap rejects narrow (0.003) and passes full (0.088). `phases-prior` reads like it (overlap on the full band, 0.034, differs), and `loud-floor` equals it (no source reaches its SNR 18.4 cap). Compare a mock's p to `fisher`'s, not to 0.05.

Marginal ranks, TARP: [../calibration/](../calibration/). Measured power of each test: [../test-power/](../test-power/).

## What the page shows

[data-space-checks.html](data-space-checks.html). The top bar picks the setting (the mock flows of the narrow or full band, with a mock picker; runs are added the same way) and, in `#setting=...&mock=...&test=...`, lives in the URL hash.

1. **Which tests reject which defect?** A ladder of γ-test p (rows: overlap, own SNR, joint log-likelihood, then the marginal and geometric tests for contrast; columns: the band's seven mocks and the runs).
2. **Ranks by SNR stratum.** For the chosen test and slot, five panels (all, then the repo's `SNR_BANDS`): ECDF minus the diagonal inside the simultaneous 95% band, a 16-bin rank histogram, n and p.
3. **The distributions behind the ranks.** Truth against the draws' median for the overlap, the own SNR (which is also snr_hat against the true SNR) and the log-likelihood.

## Method

`scripts/data_space_checks.py` reads each eval dir's `scorecard.npz`: ranks and strata from `validate.summarise` and the raw quantities and takes `gamma_test`, `SNR_BANDS` and `COORDS` from `canna.lisa.validate` unchanged. Strata are the draws' median SNR (`snr_hat` per slot, `joint_snr_hat` for the joint rank), never the true SNR.

- Slot rows keep each slot's own strata; "pooled" concatenates the 4 slots. Ladder p is over all injections, pooled over slots for per-slot tests, as in mock-flows.
- "8 coordinates, smallest p" is the smallest pooled p of the eight coordinate ranks, uncorrected for taking a minimum.
- Raw panels use the first 300 injections (all slots) and the median of the 127 draws.
- Strata with fewer than 10 ranks are not tested.

## How to rebuild

From the repo root after `source slurm/unity.env`, CPU only (the job also rebuilds the page). One setting per job:

    sbatch $CANNA_SBATCH_CPU reports/posterior-eval/data-space-checks/scripts/data_space_checks.sbatch mocks-narrow narrow reports/posterior-eval/mock-flows/data/narrow/{fisher,wide,narrow,biased,prior,phases-prior,loud-floor}
    sbatch $CANNA_SBATCH_CPU reports/posterior-eval/data-space-checks/scripts/data_space_checks.sbatch mocks-full full reports/posterior-eval/mock-flows/data/full/{fisher,wide,narrow,biased,prior,phases-prior,loud-floor}
A run is added the same way (`--smoke` marks a pipeline test):

    sbatch $CANNA_SBATCH_CPU reports/posterior-eval/data-space-checks/scripts/data_space_checks.sbatch <run>-narrow-B narrow outputs/runs/<run>/eval/<tag>

Each job takes 15 s to 1.5 min. The page alone (stdlib only): `uv run --frozen python reports/posterior-eval/data-space-checks/scripts/build_page.py`. A run's band is the `<band>` argument; a setting name containing a run name should end in the ODE steps, like the siblings.

## Data

`data/<setting>.json`: `setting`, `band`, `smoke`, `cases[]`. Each case has `name`, `mock`, `epoch`, `ode_steps`, `n`, `draws`, `ladder` (pooled p per test), `coord_p`, `panels["<test> <slot|pooled>"]` (a list over strata of `stratum`, `n`, `p`, `ecdf`, `band`, `hist`) and `scatter` (truth, draws' median pairs). Settings: `mocks-narrow` (7 mocks, 200 injections), `mocks-full` (7 mocks, 100), the old-layout `dusty-otter-narrow-XS-32` and `-128` (1000, smoke) are kept and skipped.
