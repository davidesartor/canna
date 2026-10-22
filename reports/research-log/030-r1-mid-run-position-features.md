# F30 (2026-10-11): R1 at mid-run — the position features cut the loud floor 3.5× before any cooldown; B2 launched

- Loud floor **0.0100 bins** (95% CI 0.0093–0.0110) before cooldown, passing the rule (≤
  0.012); it now falls with SNR, calibration is near nominal, no bias.
- Detection and loss lag, most likely because the position projections dominate the tokens at
  init; next runs should zero-init them.
- B2 (`B-pos`) launched 8 h early.

**Runs.** R1 = `XS-pos` (job 14263742, `PositionedLisaFlow`, 32 periods from 2 to 128 bins),
1.2M steps from scratch with XS-late-cool's recipe. Scorecard of epoch 474 (job 14275317),
constant lr, same 840 sources; `outputs/scorecards/compare_r1.py` vs XS-late-cool (same recipe
without features, cooled, epoch 1200).

| SNR | n | found, 512-cool → R1 (768-cool) | f₀ width (bins), found by both | ratio R1/512-cool [16–84%] | × ideal (R1) | in68 | in95 |
|---|---|---|---|---|---|---|---|
| < 15 | 148 | 0.50 → 0.43 (0.54) | 0.109 → 0.090 | 0.83 [0.63–1.05] | 1.8 | 0.76 → 0.72 | 0.97 → 0.93 |
| 15–40 | 157 | 0.85 → 0.77 (0.86) | 0.074 → 0.049 | 0.72 [0.49–1.09] | 2.4 | 0.92 → 0.82 | 1.00 → 0.99 |
| 40–100 | 151 | 0.95 → 0.92 (0.97) | 0.043 → 0.019 | 0.49 [0.32–0.95] | 2.4 | 0.93 → 0.74 | 0.98 → 0.97 |
| ≥ 100 | 384 | 0.995 → 0.987 (0.995) | **0.039 → 0.010** | **0.28 [0.17–0.62]** | **7** | 0.93 → 0.71 | 0.99 → 0.96 |

- **Floor:** 4.2×10⁻⁴ units of x_f; loud per-source ratio 0.283 (95% CI 0.256–0.317). The input
  representation of position was the limit. The features are computed before the bf16 cast, so
  R1 does not separate F26's candidates 1 and 2.
- **Follows SNR, as predicted:** 0.0139 / 0.0103 / 0.0084 / 0.0068 bins at SNR 100–200 /
  200–400 / 400–800 / ≥ 800 (flat 0.044–0.036 without features).
- **Calibration:** in68 0.71–0.82, in95 0.93–0.99 (baseline over-covered, loud in68 0.93). Rank
  0.48–0.52.
- **No bias:** loud offset +0.0047 ± 0.0026 bins, window-position profile ≤ 0.004.
- **Detection lags:** 31 fewer of 840 (6 only R1, 37 only 512-cool) — XS-late's level at the
  end of its constant lr (F15: 0.45 / 0.81 / 0.92 vs 0.43 / 0.77 / 0.92).

**Loss lag.**
- R1 tracks XS-late (13474448, same recipe/seed/aux phase) until epoch ~57 (0.5733 vs 0.5766 at
  55); then XS-late drops ~0.03 in 15 epochs (the F7/F10 transition) and R1 never does: 0.5240
  vs 0.4763 at epochs 373–382, gap still growing.
- At init (`outputs/scorecards/init_scale.py`) x_pos has rms 0.41 vs 0.06 for `x_embed`: a
  source token is ~98% (by variance) a random high-frequency function of its x_f, burying its
  other ten coordinates. On data tokens y_pos is 0.41 vs 1.21.
- Features are computed in float64 (global x64), not float32 as the comment says; harmless, the
  bf16 cast comes after.
- A reviewer's CPU check on 40 XS-pos windows found positions consistent: source bins match
  f0·T_obs − window_start to 5×10⁻¹¹ bins, token column c covers channels 2c, 2c + 1, the
  power-weighted token centroid sits −0.02 ± 0.37 bins from the source. Not a position bug.
- **Any next run with features should zero-initialise `x_pos` and `y_pos`.**

**B2 launched** (14275499 → 14275503, scorecard 14275504, eval 14275505).
- The rule says B reruns with features; the margin is clear (CI tops at 0.011) and every
  cooldown so far lowered the floor. Starting 8 h before R1 ends saves 8 h of an 87 h run at
  the same GPU budget.
- `configs/B-pos.yaml` (3eb92aa): B-late plus `positions: 32` (periods 2 to 8192 bins), R1's
  design with default init so results map onto R1's.
- CPU smoke test (`outputs/scorecards/b_pos_smoke.py`): 75.70M parameters, finite outputs, 2048
  distinct token positions (641 in B1).

**Predictions.**
- **R1 after cooldown:** loud floor 0.004–0.008 bins (cooldown ×0.44 in C1/C2); detection
  within ~10 sources of 512-cool; loud in68 0.68–0.80.
- **B2**, two hypotheses its scorecard separates:
  - (a) precision stays a fraction of the window: 4×10⁻⁴ of x_f is 0.6 bins on B, so loud
    widths 0.3–0.8 bins, most loud sources found;
  - (b) features make it a precision in bins set by the 2-bin periods: loud widths ~0.01–0.05
    bins, limited by B's 1/6 of the steps.
  - Either way B2 finds most sources at SNR ≥ 40 (B1: none). Early read: a scorecard at epoch
    ~60 (~27 h in, 43 min of GPU).
