# F44 (2026-10-15): RoPE breaks B's 3-bin floor: 215 of 240 sources found, 70× narrower, calibrated

- B-rope finds 215 of 240 sources (< 1 bin; B-fixed 5), 50–80× narrower than B-fixed above SNR
  15, calibrated.
- B's ~3-bin floor was the sources finding their tokens; a new floor sits at ~0.035 bins.
- The held-out loss moves 0.3% for this 70× change (F6).

**Run.** B-rope (14509895, B-fixed + RoPE in frequency, F42): 11 h 12 min with scorecards
(`--n_random 50`, same 240 sources as B1) and eval, 189 s per epoch (B-fixed 178 s), peak 5.5
GiB. The watcher saved the full final checkpoint with `opt_state` (so x). Local as
`outputs/18-lisa-B-rope`. Tables: `outputs/scorecards/compare_fixed.py`.

**B, the same 240 sources** (median f₀ width in bins, all sources, by SNR band; per source vs
B1):

| model | < 15 | 15–40 | 40–100 | ≥ 100 | found (< 1 bin) | < 2 bins | per source vs B1 | rank / in68 / in95 |
|---|---|---|---|---|---|---|---|---|
| B1 (85 h) | 4.97 | 3.80 | 2.93 | 3.17 | 1 | 2 | 1 | 0.44 / 0.88 / 0.99 |
| B-fixed x | 3.77 | 2.74 | 2.54 | 2.81 | 5 | 43 | 0.69 / 0.67 / 0.77 / 0.90 | 0.53 / 0.93 / 0.99 |
| **B-rope x** | **0.91** | **0.054** | **0.035** | **0.036** | **215** | **219** | **0.06 / 0.01 / 0.01 / 0.01** | 0.51 / 0.88 / 1.00 |
| B-rope y | 0.89 | 0.047 | 0.030 | 0.031 | 214 | | | 0.52 / 0.82 / 0.99 |

- **Per source vs B-fixed** (x): 0.083 [95% CI 0.040–0.248] below SNR 15, 0.021 [0.016–0.026]
  at 15–40, 0.014 at 40–100, **0.012 [0.011–0.014] at ≥ 100**.
- **Calibrated:** rank 0.51, in95 1.00, offsets a few hundredths of a bin, either sign.
  Slightly conservative at 68% (0.88).
- **New floor ~0.035 bins**, flat from SNR ~40 up (loud 0.036, 95% CI 0.030–0.038). In Fisher
  widths: 2–6× at SNR 15–100, 12× at 100–300, 32× at 300–1000, 72× above 1000. 3.4× XS-rope's
  loud floor (0.0106), 5× R1's (0.0067).
- **Eval draws** (10 injections, 40 sources; nearest slot to each true source):
  - median source: 96% of draws within 1 bin, 92% within 0.2 bins (B-fixed 26% and 5%);
  - far modes nearly gone: 2 of 40 sources keep > 5% of draws beyond 10 bins (B-fixed 16), both
    SNR 7–8;
  - F41's bimodal loudest source (SNR 1668) is now clean;
  - misses: injection 5's SNR-44 source sits 1–10 bins off; two of injection 7's faint sources
    (SNR 8.0, 7.7) are wide or off.
- **Loss:** held-out x 0.6399 vs B-fixed's 0.6417. RoPE led by 0.024 at epoch 50 and the gap
  closed; the widths did not.
- **Still training:** held-out x fell 0.0026 over the last 25 epochs (0.6425 → 0.6399).

**What B's floor was.** Flat in SNR across B1, B-sfplus and B-fixed, unmoved by float32 (F42).
- Not the f₀ PLE bin, nor the bf16 cast: both are unchanged in B-rope.
- It was the sources finding their tokens: 256 data tokens with learned absolute positions vs a
  source coordinate in log f. RoPE makes each score a function of f₀ − f_token.

XS (4 tokens) had little to find, hence 0.63× there, not 70× (F43).
