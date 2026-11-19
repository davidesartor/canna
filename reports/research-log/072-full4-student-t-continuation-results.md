# F72 (2026-11-19): the Student-t continuation of full S=4 (261103-n4-full-B-t44) trained to epoch 64 with every loss finite and is the best full-band model on every parameter; every full S=4 entry of the paper filled

**Outcome.**
- **Training** (TREX 18085060, 20 h 32 min, epochs 45-64 at ~3690 s each, one Slurm segment): every loss finite.
  Largest single-step gradient norm 3.5e4-4e5 per epoch through epoch 61, 3.3e6 at epoch 62 and 1.3e6 at 64,
  against 7e9 -> 9e17 in epochs 48-50 of the Gaussian run. The epoch-mean gradient norm rose in the cooldown
  (55-80 -> 346), as in the healthy full S=8 run (53 -> 274, peaks 5e5): a cooldown pattern, not a new instability.
  The logged frame NLL (Gaussian, with k^2) rose 0.73 -> 1.81 while the t-frame kept narrowing (bin-mean
  log-width -0.22 before the switch, -0.44 at epoch 45, -0.52 at 64); the catalogue flow NLL sat at -0.50..-0.61.
  So the best-checkpoint rule stayed at epoch 45 and the eval read the last checkpoint (18162618).
- **Widths against the joint MCMC** (flow / chains, median per SNR band; 261103-vs-mcmc-t44.json):

  | model | SNR | f0 | log fdot | log A | lon | lat | sin iota | found |
  |---|---|---|---|---|---|---|---|---|
  | t44 (epoch 64) | 7-15 | 1.66 | 1.07 | 1.21 | 1.57 | 1.39 | 1.21 | 0.62 |
  | | 15-40 | 1.22 | 1.19 | 1.17 | 1.84 | 1.65 | 1.22 | 0.77 |
  | | >=40 | 6.61 | 5.75 | 1.56 | 5.00 | 3.70 | 1.60 | 0.82 |
  | e32 fork (epoch 40, Gaussian) | 15-40 | 1.48 | 1.24 | 1.23 | 1.90 | 1.79 | 1.21 | 0.71 |
  | | >=40 | 9.74 | 7.66 | 1.72 | 7.03 | 4.14 | 1.60 | 0.81 |
  | crashed main (best, epoch 43) | >=40 | 17.2 | 13.3 | 2.62 | 13.6 | 8.20 | 2.29 | 0.67 |
  | full S=8 (epoch 64) | >=40 | 10.7 | 9.05 | 1.97 | 8.61 | 5.79 | 1.92 | 0.66 |

  At SNR 7-15 the f0 width is 0.96x the Fisher width; coverage after relabelling 0.68-0.72 (68%), 0.91-0.95 (95%).
  Off the chains' peak: 28% of loud draws (full S=8 40%, S=16 48%; chains 2%).
- **Calibration without the truth**: SBC 68% coverage 0.691 (MCMC 0.694), smallest gamma-test p 0.016 (Im z-;
  MCMC 0.078 on fdot). Validation pass (scorecard 18292006, 7 min): coordinate min p 0.016, TARP 0.013, and
  overlap and joint log-likelihood p < 0.001 as for every flow; drawn overlap offset +0.20 (mock +0.05).
- **JS divergence** (import_js.py main): mean 0.19 at SNR 7-15 and 0.29 at >= 15 (full S=8 0.21/0.36, S=16
  0.28/0.40; narrow 0.08-0.17); median per coordinate 0.016-0.039, f0 0.023, the lowest f0 of all six problems.
- **Frame or flow** (frame_vs_flow.py on t44, TREX 18292007; outputs/runs/261103-analysis/frame-vs-flow-f4.npz,
  the e32 one renamed -f4-e32): frame / chains 0.83, 15.2, 23.9, 62.0 at SNR < 15, 15-40, >= 40 below and above
  10 mHz, against 1.04, 28.7, 53.8, 121 for e32 on the same problem: the t-frame is about half as wide on loud
  sources (F71's 0.6x), and the flow ends narrower too (1.24 / 1.89 / 20.1 vs 1.52 / 2.30 / 27.9). Above 10 mHz
  the flow still narrows its frame only ~3-fold (14-fold below): F68's high-frequency gap remains.
- **Overlap groups** (overlap_groups_flow.py, 18292007): flow offset +0.201 drawn, +0.256 frequencies, +0.233 sky,
  +0.270 phases, +0.214 moduli, +0.208 frequencies+phases; mock +0.054 drawn, +0.04-0.05 frequencies/sky,
  +0.13 phases: F69's pattern on the sixth problem.
- **Faint sources** (SNR < 3): flow 0.95x, chains 0.97x the prior's f0 width.

**What it does not show.** t44 differs from every other full-band model in three ways at once: the Student-t
frame, the fewest sources (S=4) and, against e32, 24 more epochs. One run, one seed: the gain over e32 is not
attributable to the Student-t loss, and whether it helps the other five problems is open. The comparison that
would isolate it (Gaussian full S=4 to epoch 64) is the run that diverged.

**Paper (2026-11-19).** import_runs.py (`2026-11-11 n4-full-main t44`, with a fallback to e64.reference_for when
load_reference returns None under the current code; every other run's entry unchanged), import_js.py main,
import_sbc/faint/corners/ranks/overlap/frame.py, numbers_results.py, fig_results/corners/rank-ecdf: every full
S=4 ?? and the summary row filled; red TODOs on t44 in sections/experiments.tex and appendix/results.tex
removed; the Student-t frame loss described in app:fm-frame, the continuation in app:training-optimiser
(\ResultsTFourMaxGrad, \ResultsTFourGaussMaxGrad from the histories); app:results-runs says why the last
checkpoint is read; the full-band widths and corner text updated for t44 (the S=4 -> 8 growth is now 50-70%
in f0, fdot and sky, not "10 to 40 per cent").

**Open.** The best-checkpoint rule (catalog frame + flow, Gaussian k^2) must use flow + 2 delta_s before any
further frame_nu run. TREX: run, logs and analyses fetched and md5-verified (outputs/trex-t44-261119/), then
deleted; the local run is outputs/runs/261103-n4-full-B-t44 (numbered 41-lisa-full-B-n4-t44).
