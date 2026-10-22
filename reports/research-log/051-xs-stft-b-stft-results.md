# F51 (2026-10-18): STFT tokens narrow loud sources' angles and amplitude 0.6–0.85×; φ₀ still not learned, not the AdamW decay

- B-stft: loud-source A, sky, ι, ψ 0.6–0.85× B-rope's, no cost below SNR 100, f₀ unchanged;
  equals B-long at a third of the steps.
- XS-stft: loud sky narrows, but f₀ regresses 13–23% below SNR 300.
- φ₀ is unlearned with no head decay, so F48's mechanism is ruled out.

**Runs.** XS-stft 14891481 (11 h 28 min, 33 s/epoch, 2.0 GiB), B-stft 14891484 (11 h 25 min,
203 s/epoch vs B-rope 189, 7.4 GiB); post 14891483, 14891486 at 24e791b (pages save samples,
fold ψ/φ₀). Outputs `22-lisa-B-stft`, `23-lisa-XS-stft`. Held-out x loss vs rope: B 0.6334 vs
0.6399 (−1.0%), XS 0.7361 vs 0.7294 (+0.9%). Tools: `outputs/scorecards/compare_stft.py`,
`reports/diagnostics/draw-widths/plot_widths.py at 09abb90` (width ratios),
[`velocity_probe.py`](../posterior-eval/flow-vs-reference/scripts/velocity_probe.py).

**f₀ (scorecards; per-source width over the rope run, median [95% CI], by SNR).**

| run | found | 0–15 | 15–40 | 40–100 | 100–300 | ≥ 300 | in68 / in95 |
|---|---|---|---|---|---|---|---|
| B-rope | 215 / 240 | 1 | 1 | 1 | 1 | 1 | 0.88 / 1.00 |
| B-long (600k) | 221 | 0.75 | 0.88 | 0.80 | 0.85 | 0.79 | 0.86 / 1.00 |
| **B-stft** | **217** | 0.92 [0.68–1.19] | 1.03 [0.80–1.16] | 0.87 [0.79–1.01] | 1.00 [0.84–1.13] | 0.87 [0.75–0.99] | 0.85 / 0.99 |
| XS-rope | 688 / 840 | 1 | 1 | 1 | 1 | 1 | 0.75 / 0.97 |
| **XS-stft** | **658** | 1.19 [1.14–1.32] | 1.22 [1.15–1.30] | 1.13 [1.08–1.20] | 1.23 [1.11–1.31] | 1.05 [1.01–1.11] | 0.73 / 0.97 |

XS lost 41 sources and gained 11.

**Every parameter (40 eval sources; width over the reference run, median [16–84%]; ecliptic).**

| | SNR | n | f₀ | A | λ | sin β | ψ | sin ι | φ₀ |
|---|---|---|---|---|---|---|---|---|---|
| B-stft / B-rope | < 15 | 6 | 0.97 | 0.93 | 1.14 | 1.05 | 1.17 | 0.93 | 0.99 |
| | 15–100 | 20 | 0.88 | 0.94 | 1.01 | 0.97 | 0.97 | 0.95 | 1.00 |
| | 100–300 | 4 | 1.29 | **0.77** | **0.75** | 0.85 | 0.95 | 0.85 | 1.00 |
| | ≥ 300 | 10 | 1.04 | **0.69** [0.55–0.88] | **0.62** [0.36–0.89] | **0.58** [0.29–0.90] | 0.83 | **0.74** | 1.00 |
| B-stft / B-long | 15–100 | 20 | 0.96 | 1.02 | 1.21 | 1.17 | 1.15 | 1.12 | 1.00 |
| | ≥ 300 | 10 | 1.21 | 1.01 | 0.97 | 0.93 | 0.94 | 1.11 | 1.00 |
| XS-stft / XS-rope | 15–100 | 19 | 1.22 | 1.14 | 1.10 | 1.13 | 1.03 | 1.02 | 1.00 |
| | 100–300 | 5 | 1.38 | 0.87 | 1.17 | 1.08 | 0.86 | 0.90 | 1.00 |
| | ≥ 300 | 10 | 0.99 | **0.80** | **0.73** | **0.67** | 0.90 | 0.87 | 1.01 |

**Against the MCMC (flow / MCMC width).**
- B loudest (SNR 1668): sky 66, 106 (B-rope) → 30, 44 (B-long) → **18, 21** (B-stft); W1 58,
  124 → 20, 23. f₀ width 210× vs 56×, but W1 2066 vs 11532 (fewer strays).
- XS loudest (SNR 1366): sky 15.6, 16.5 → **7.4, 7.3**; f₀ 28 → 14.
- B mid SNR (17–78): within the rope run's noise; A, ψ, ι 0.9–1.6×.
- Worse: XS gb-median (SNR 62) sky 11.5, 6.0 → 24.8, 14.9, f₀ 7.6 → 24; B faintest (SNR 7.3) λ
  5.4 → 13×.
- φ₀: XS 61, 46, 30×, B 47× where the MCMC pins it, as in every run since F39.

**φ₀ (velocity probe, final checkpoints).**
- Velocity spread 0.3% (XS) and 0.5% (B) of target, R² 0.00 at every s; head error equals
  x_t's; head rows 0.06× (XS) and 0.13× (B) the median row (XS 0.19× at epoch 77).
- No decay on these rows, so they shrank because the network never predicts φ₀, as an output
  with an unpredictable target does. Small rows are the symptom, not the cause: **F48's
  mechanism is ruled out**.
- The representation is not it either: STFT tokens, giving φ₀ as an intercept, change nothing.
- Still separating XS-fd (learned φ₀) from every run since F39: 0.05 decay on z for Muon
  matrices (XS-fd 1e-6), 16-bin tokens (XS-fd 2), SNR-ordered slots, the reconstruction head,
  NorMuon.

**F50's predictions.**
1. φ₀ moved by epoch ~50: **no** (0.6% at epoch 77, 0.3% final).
2. ψ within 1–2× on loud sources: **no** (B loudest 2.45×, B-rope 2.32; XS 10×, was 10.4).
3. Loud sky narrows: **yes on both** (2–5× vs MCMC on the loudest, 0.6–0.7× per source above
   SNR 300). Against F49's pure bf16-cast floor: the finer input encoding moved it.
4. f₀ and found at least the rope runs': **B yes, XS no.**

**Reading.**
- XS (4 tokens) loses at mid SNR, plausibly because per-sample |z| ~1–5 makes the gated phase
  mostly noise and the log-polar features nonlinear in the noise, with little context to
  average.
- φ₀ needs a different fix: resample (A, ι, ψ, φ₀) per draw from their exact likelihood given
  (f₀, ḟ, sky) via the four F-statistic amplitudes (training-free); or one XS run with no decay
  on z, the last optimiser difference from XS-fd (careful of F36's z runaway).
