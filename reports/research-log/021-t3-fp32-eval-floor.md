# F21 (2026-10-05): T3' — fp32 at eval leaves the floor where it is; the cooldown is lowering the loss
- fp32/bf16 loud-source width ratio at eval: **0.991** (95% bootstrap 0.978–1.000, 93 sources).
  The decision rule was ≥ 0.9, so **bf16 at eval is ruled out**; the bf16-noise prediction ≈
  0.56 is excluded, and F5's 6% was noise.
- The C1 cooldown has already cut the flow loss by 0.012: optimiser noise at the constant lr
  was holding the loss up.

**T3' (job 13780225):** XS-late, fp32 on a CPU node, 1 h 15 min; the 10 eval plus the first 40
random injections (200 sources), same keys as the bf16 A100 scorecard (F15):

| SNR | n | found, bf16 / fp32 | f₀ width, bf16 → fp32 (bins) | width ratio fp32 / bf16, median [16–84%] |
|---|---|---|---|---|
| < 15 | 33 | 0.39 / 0.39 | 0.154 → 0.141 | 0.98 [0.92–1.00] |
| 15–40 | 41 | 0.78 / 0.78 | 0.132 → 0.131 | 0.99 [0.93–1.03] |
| 40–100 | 31 | 0.94 / 0.94 | 0.098 → 0.093 | 0.99 [0.95–1.03] |
| ≥ 100 | 95 | 1.00 / 1.00 | 0.088 → 0.088 | 0.99 [0.95–1.04] |

- Both find exactly the same 169 sources.
- **fp32 shifts every loud median by −0.026 ± 0.002 bins** (~1/4 width), about uniformly across
  the window (a little more at the top) and independent of the input's bf16 cell, so not input
  rounding. Coverage unchanged (in68 0.86 vs 0.85, in95 1.00). The bf16 medians are a little
  closer to the truth (median |offset| 0.038 vs 0.053 bins): trained in bf16, that is the
  network's best eval precision.
- **Still open:** bf16 *in training* (F5, Open): one costlier run with an fp32 head, after C1.
- **Speed:** 88 s per injection on 64 cores vs ~1.6 s on an A100.

**C1 interim (job 13780234, epoch 1118 of 1200).** 10-epoch mean flow loss **0.4499** (epochs
991–1000, constant lr) → 0.4490, 0.4471, 0.4450, 0.4422, 0.4400, **0.4377** (1101–1110, lr at
0.45). −0.012 so far already beats the −0.004 to −0.010 predicted for the whole cooldown, ~10×
faster than the last 100k constant-lr steps (−0.0015). Whether noise also holds the floor up
waits for the scorecard.
