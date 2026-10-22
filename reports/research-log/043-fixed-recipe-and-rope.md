# F43 (2026-10-14): RoPE brings XS's loud f₀ back to XS-fd's level; B-fixed is the best B so far; the 1/L1 weighting was not XS's regression

- **XS:** with 1/L1 off, XS-fixed still lands on XS-sfplus. RoPE returns the loud floor to
  XS-fd's (0.0106 vs 0.0103 bins), but finds 45 fewer sources and is 1.25–1.35× XS-fd below SNR
  100.
- **B:** B-fixed is 10–33% narrower than B1 per source, calibrated, but the floor is still flat
  at ~2.8 bins (rule < 1 bin fails).
- The averaged weights x were not kept in the final checkpoints; only y.

**Runs** (F42's launches): XS-fixed (14509892), B-fixed (14509893), XS-rope (14509894)
completed with scorecards (x and y) and eval draws; B-rope (14509895) still running. Fetched as
`outputs/15-lisa-XS-fixed`, `16-lisa-B-fixed`, `17-lisa-XS-rope`. Tables:
`outputs/scorecards/compare_fixed.py` (sources matched by injection and SNR) and
`lisa_checks/mcmc_table.py at 05393cd` (`outputs/mcmc/compare_{fixed,rope}_x.log`,
`table_fixed_rope.txt`).

**XS, the same 840 sources** (median f₀ width in bins of the found, by SNR band; per source vs
XS-fd's x):

| model | found | < 15 | 15–40 | 40–100 | ≥ 100 (95% CI) | rank / in95 | per source vs XS-fd |
|---|---|---|---|---|---|---|---|
| R1 | 739 | 0.098 | 0.044 | 0.016 | 0.0067 | 0.51 / 0.97 | 1.08 / 1.00 / 0.94 / 0.65 |
| XS-fd x | 732 | 0.080 | 0.043 | 0.017 | 0.0103 | 0.45 / 0.96 | 1 |
| XS-sfplus x | 681 | 0.113 | 0.059 | 0.026 | 0.0151 | 0.50 / 0.98 | 1.61 / 1.55 / 1.62 / 1.50 |
| XS-fixed x | 678 | 0.109 | 0.051 | 0.022 | 0.0149 (0.0141–0.0165) | 0.51 / 0.98 | 1.44 / 1.45 / 1.51 / 1.54 |
| **XS-rope x** | **687** | 0.102 | 0.059 | 0.022 | **0.0106** (0.0094–0.0121) | 0.51 / 0.97 | 1.25 / 1.35 / 1.29 / **1.08** |
| XS-fixed y | 664 | | | | 0.0179 | 0.50 / 0.96 | 1.66–1.77 |
| XS-rope y | 659 | | | | 0.0116 | 0.50 / 0.97 | 1.16–1.43 |

- **1/L1 was not the regression.** Of F40's suspects, the 4 tokens, the SNR-ordered labels and
  the weight shrinkage (below) remain.
- **RoPE vs XS-fixed, per source:** 0.88 [0.85–0.90] at SNR 15–100, 0.72 [0.67–0.75] at
  100–400, **0.63 [0.60–0.66] at ≥ 400**. Calibration unchanged.
- **The loss does not see it** (F6 again): held-out x 0.7294 with RoPE, 0.7283 without.
- **x beats y** in both runs, as in F40.

**Against the MCMC** (x, four windows, median width ratio to MCMC):

| | R1 | XS-fd x | XS-sfplus x | XS-fixed x | XS-rope x |
|---|---|---|---|---|---|
| loud f₀ (SNR ≥ 500) | 6.3 | 11.4 | 21.7 | 24.1 | **11.4** |
| loud A, sky, orientation | 5.0 | 6.1 | 7.2 | 7.4 | 7.5 |
| f₀ at SNR 50–500 | 2.7 | 3.7 | 6.3 | 5.0 | 5.2 |
| MCMC median inside the 95% | 125/128 | 126/128 | 127/128 | 123/128 | 125/128 |
| sources missed, of 16 | 6 | 6 | 8 | 8 | **6** |

RoPE recovers window 9's SNR-33 source (f₀ 11× vs 170× for XS-fixed; XS-fd 1.7×) and halves
window 9's loudest pair (28× and 8× vs 52× and 31×). A, sky and orientation do not move: RoPE
places a source in frequency, not on the sky.

**B, the same 240 sources** (median f₀ width in bins, all sources; per source vs B1):

| model | < 15 | 15–40 | 40–100 | ≥ 100 | < 2 bins | per source vs B1, by band | rank / in95 |
|---|---|---|---|---|---|---|---|
| B1 (85 h) | 4.97 | 3.80 | 2.93 | 3.17 | 2 | 1 | 0.44 / 0.99 |
| B-sfplus x | 4.39 | 3.38 | 3.01 | 3.24 | 25 | 0.88 / 0.81 / 0.99 / 1.03 | 0.52 / 1.00 |
| **B-fixed x** | **3.77** | **2.74** | **2.54** | **2.81** | **43** | **0.69 / 0.67 / 0.77 / 0.90** | 0.53 / 0.99 |
| B-fixed y | 4.18 | 2.66 | 2.24 | 2.38 | | 0.86 / 0.65 / 0.73 / 0.72 | **0.41** / 0.95 |

- B-fixed finds 5 sources (B1 1). Floor slope in SNR −0.002.
- **On B, 1/L1 off helped** (the only change vs B-sfplus); on XS it did not. One seed each.
- **y is now biased the other way** (rank 0.41; B-sfplus's y 0.62): x stays the model.

**Training** (final epoch held-out x): 0.7283 (XS-fixed), 0.7294 (XS-rope), 0.6417 (B-fixed;
B-sfplus 0.6451).

**Correction to F40: the weights shrink without 1/L1 too.** Backbone RMS falls from 0.019 to
0.0081 (XS-fixed; XS-sfplus 0.0070) and 0.0110 (B-fixed; B-sfplus 0.0118); F40's predicted
equilibrium near 0.020 was wrong. The decay on z (0.05) does it; gradient norm grows to 4.6
(XS) and 10.9 (B), so the clip at 1 binds throughout. What 1/L1 added was mostly the gradient
growth (8.8 on XS-sfplus).

**x not kept.** `lisa-local.sbatch` deletes `opt_state` (where schedule-free stores x) before
the copy home, so the final checkpoints of XS-fixed, XS-rope and B-fixed hold y only. Their
in-job x scorecards and eval draws are kept, and mid-run snapshots hold x at epochs 839, 827
and 156. For B-rope, a watcher step (`srun --overlap`, log `.slurm-logs/b-rope-keep-x.log`)
copies the full final checkpoint to scratch (`outputs/lisa-B-rope-x`) on `[done]`, before
cleanup.

**Storage:**
- TREX `~/canna/outputs` is now a symlink to `/work/scratch/data/mentasgi/canna-outputs`, off
  the home quota; a failed copy home would have run the EXIT trap and deleted the run.
- Mid-run snapshots (with `opt_state`) are local in `outputs/trex-snapshots/` (all 212 arrays
  finite).
