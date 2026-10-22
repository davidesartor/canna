# F40 (2026-10-13): XS-sfplus — schedule-free now works as a mechanism, but the run regressed on every comparable metric

- The averaging is fixed: z − x stays bounded and x beats y.
- But the optimizer drove the weights down 3.6× below init, the clip binds permanently, and the
  run is ~1.5× wider than XS-fd with fewer detections.
- The cause is not separable: XS-sfplus changed six things at once.

**Runs.** XS-sfplus (14410743) and B-sfplus (14410744), F39's recipe, both completed on TREX.
Scored locally from XS's y only (`lisa_checks/scorecard_weights.py at 05393cd`, 840 sources;
`lisa_checks/compare_mcmc.py --tag _sfplus_y`, four windows, now
[`compare_joint.py`](../posterior-eval/flow-vs-reference/scripts/compare_joint.py)); XS's x, the TREX scorecards and corner pages,
and all of B were not yet fetched. `compare_mcmc.py` now relabels cached Fisher draws onto the
truth (the loudest-first ordering broke it).

**Mechanism fixed.** z − x bounded at 0.013 RMS from epoch ~75 (XS-fd ended at 0.49). Held-out
x − y = −0.005 to −0.009 from epoch 200, growing (XS-fd ended at +0.012).

**But the weights shrank.**

| epoch | backbone weight RMS | gradient norm (pre-clip) |
|---|---|---|
| 48 (end of warm start) | 0.027 | 0.29 |
| 200 | 0.015 | 2.4 |
| 1200 | **0.0070** | **8.8** |

- The clip at 1 binds from epoch ~100: behind normalisations the gradient grows as 1/|w|.
- **The loop:** the 1/L1 weighting reads the *clipped* gradient, whose L1 rises over the
  warm-start reference, so the step shrinks. Decay was tied to the same weighted lr, so the
  equilibrium norm ≈ 0.2·√(s·lr/2λ) falls with the weighting factor s, and the gradient grows
  further. Without the weighting, λ = 0.05 at lr 1e-3 balances near 0.020 (initial 0.0255).
- ScheduleFree+ pairs its adaptive step with fully decoupled decay ∝ γ² (equilibrium
  independent of γ) and warns the coupled form "WILL break learning rate adaptation". F39's
  integration was wrong on this point.
- **No late loss drop.** The effective (angular) step grows ~4× over the run, the opposite of
  annealing. In R1 and XS-fd the drop coincided with growing weights (F34), supporting it as
  their weight growth's implicit lr decay.

**Scorecard of y** (same 840 sources as F33):

| SNR | found: R1 / XS-fd (x) / **XS-sfplus (y)** | f₀ width per source, XS-sfplus / XS-fd [16–84%] | in95 | rank |
|---|---|---|---|---|
| < 15 | 0.53 / 0.50 / **0.35** | 1.67 [1.07–2.98] | 0.92 | 0.49 |
| 15–40 | 0.85 / 0.84 / **0.69** | 1.46 [1.13–2.66] | 0.98 | 0.50 |
| 40–100 | 0.96 / 0.95 / **0.89** | 1.58 [1.17–2.42] | 0.98 | 0.50 |
| ≥ 100 | 0.995 / 0.997 / 0.98 | 1.54 [1.07–2.25] | 0.95 | 0.50 |

- Loud floor 0.015 bins (12× ideal), against XS-fd's 0.0103 and R1's 0.0067.
- 673 of 840 found, against 732 and 740 (67 only by XS-fd, 8 only by XS-sfplus).
- Calibration good: honest but wider.

**Against the MCMC** (y, four windows, median ratio to MCMC width):

| | R1 | XS-fd x | XS-fd y | XS-sfplus y |
|---|---|---|---|---|
| loud f₀ (SNR ≥ 500) | 6.3 | 11.4 | 13.8 | **20.4** |
| loud A, sky, orientation | 5.0 | 6.1 | 5.6 | 7.5 |
| f₀ at SNR 50–500 | 2.7 | 3.7 | 3.6 | 6.4 |
| MCMC median inside the 95% | 125/128 | 126/128 | 125/128 | 127/128 |
| sources missed, of 16 | 6 | 6 | 5 | 7 |

The new miss is window 9's SNR-33 source, found by every earlier model. Window 9's SNR-14
source is better than ever (f₀ 15× against 27–45×; sky 1.1–1.7× against 6–10×).

**Candidate causes** (six changes against XS-fd, F34, F39). Three can each plausibly widen
everything:
1. 4 tokens instead of 32: ~8× less backbone work on the data, close pairs share a token.
2. SNR-ordered labels: 38% longer paths (F34), near-equal faint sources swap labels.
3. The optimizer regime: weights 3.6× below init, rising effective step, binding clip.

NorMuon, the reconstruction head (weight 0.1) and the averaging fixes are less likely to cost
1.5× everywhere; the averaging fixes demonstrably helped.

**B-sfplus** shows the same pattern: losses 0.669 training, x 0.645; weights 0.0118, gradient
8.4, z − x 0.021. Scorecards on TREX, not yet read.
