# F15 (2026-10-03): scorecards — the warp narrows posteriors 1.6× and removes the f₀ bias
- XS-late is ~0.6× as wide as the 1M uniform-clock model, unbiased and still conservative;
  detection is slightly worse (net −21 of 840).
- The floor moved to 0.088 bins (~60× ideal at SNR ≥ 100) but is still a floor, and not bf16.
  **The warp is a keeper, but not enough on its own.**

Jobs 13626493 (1M, uniform clock) and 13626496 (XS-late), ~4 min each on an A100; 840 sources
(10 eval + 200 random prior injections); outputs in `outputs/scorecards/`. found = f₀ width < 1
bin; ideal width = √3/(π SNR) bins; rank = fraction of draws below the truth (0.5 unbiased);
in68/in95 = fraction of truths inside the flow's central intervals.

| SNR | found 1M → late | f₀ width (bins) 1M → late | × ideal 1M → late | median offset (bins) 1M → late | rank 1M → late | in68 / in95, 1M → late |
|---|---|---|---|---|---|---|
| < 15 | 0.52 → 0.45 | 0.223 → 0.152 | 5 → 3 | −0.085 → +0.010 | 0.56 → 0.50 | 0.80 / 0.98 → 0.78 / 0.99 |
| 15–40 | 0.85 → 0.81 | 0.187 → 0.129 | 10 → 6 | −0.042 → +0.021 | 0.58 → 0.46 | 0.87 / 0.99 → 0.95 / 1.00 |
| 40–100 | 0.95 → 0.92 | 0.151 → 0.088 | 18 → 11 | −0.053 → +0.013 | 0.60 → 0.47 | 0.80 / 0.98 → 0.89 / 0.99 |
| ≥ 100 | 0.99 → 0.99 | 0.147 → 0.088 | 94 → 59 | −0.056 → −0.003 | 0.62 → 0.50 | 0.77 / 0.97 → 0.87 / 0.99 |

- **Narrower:** on sources both find, the median late/1M width ratio is 0.61 at SNR ≥ 100
  (narrower in 97%), 0.62 at 40–100, 0.70 at 15–40.
- **F12's low f₀ bias is real in 1M and gone in XS-late.** 1M ranks of 0.56–0.62 are ~8σ from
  0.5 for the 369 loud random sources; XS-late's are 0.46–0.51, offsets ≈ 0. Both are
  conservative (in95 0.97–1.00, in68 above 0.68): no over-confidence.
- **Detection:** 27 found only by 1M (median SNR 17), 6 only by XS-late. The expected warp
  cost: ~21% of training samples land in the first half of the path, vs 50% uniform. Part may
  be the eval: with uniform steps in t, s < 0.5 gets only ~6 of the 32 RK4 steps.
- **Floor:** about flat in SNR (Spearman −0.16). Not bf16: it does not track the f₀
  coordinate's bf16 cell (Spearman −0.11, the wrong sign); 0.023-bin cells give 0.089 bins,
  0.094-bin cells 0.086.
- **The A100 reproduces the CPU table** (F12) on the 31 localised eval sources: median width
  ratio 0.97, one outlier (the collapsed q0.40 source). F14's laptop problem does not reach
  TREX.
- **E2's decision rule:** the floor fell 1.7×, not ≥ 3× (0.088 bins, not ≲ 0.06); calibration
  improved; B fell slightly.
