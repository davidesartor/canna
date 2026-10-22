# F19 (2026-10-05): E3' — capacity does not move the floor; it wins back half the detection
- 2.25× the parameters and a 1.3% lower loss leave the floor the same within 1%: capacity is
  ruled out. Net +11 sources found vs XS-late at 1.5× the cost; calibration similar.
- Loud-source f₀ is biased by ~1/3 of a width, with a run-specific shape. An lr cooldown or
  weight EMA is the standard cure, and may also be what holds the floor up.

**Training (job 13719349):** 16 h 36 min, 59.6 s per epoch, no NaN. Final flow loss **0.4498**
vs XS-late's 0.4557; the gap was 0.023 at epoch 50 (the drop came sooner), 0.004–0.008 from
epoch 300 on; end slope −0.0015 vs −0.0013 per 100 epochs. The y aux metric grows without bound
at weight 0 (521; 1.3×10⁶ in XS-late): harmless, as that head is untrained and sampling never
uses it.

**Scorecard (job 13767819, 6 min), corners (13767924, 14 min)** on F15's 840 sources; widths
are medians over sources both models found:

| SNR | n | found, 1M / late / 768 | f₀ width, late → 768 (bins) | per-source ratio 768 / late, median [16–84%] | in68, late → 768 | in95 | rank |
|---|---|---|---|---|---|---|---|
| < 15 | 148 | 0.52 / 0.45 / 0.47 | 0.149 → 0.158 | 1.00 [0.68–1.28] | 0.78 → 0.80 | 0.99 → 0.97 | 0.50 → 0.49 |
| 15–40 | 157 | 0.85 / 0.81 / 0.83 | 0.128 → 0.117 | 0.92 [0.70–1.18] | 0.95 → 0.87 | 1.00 → 0.99 | 0.46 → 0.48 |
| 40–100 | 151 | 0.95 / 0.92 / 0.94 | 0.088 → 0.089 | 1.00 [0.77–1.17] | 0.89 → 0.81 | 0.99 → 0.98 | 0.47 → 0.47 |
| ≥ 100 | 384 | 0.99 / 0.99 / 0.995 | 0.088 → 0.087 | 0.96 [0.80–1.17] | 0.87 → 0.82 | 0.99 → 0.98 | 0.50 → 0.41 |

- **Floor:** 0.087 / 0.087 / 0.085 / 0.090 bins at SNR 100–200 / 200–400 / 400–800 / ≥ 800
  (XS-late 0.089 / 0.091 / 0.087 / 0.087). With capacity added to F16's list (integrator, input
  rounding, A100 numerics, physics), two candidates remain:
  - **bf16 inside the network** (activations, output): a floor identical across widths is how a
    fixed resolution behaves; 2⁻⁸ ≈ 0.004 flow units = 0.09 bins (1 unit = 24 bins), the
    floor's size. T3 was to test this; still void (F18).
  - **Optimiser noise at a constant lr** (F9); see the bias below.
- **Detection:** 16 found only by 768 (median SNR 16), 5 only by XS-late: net +11 of 840 (sign
  test p = 0.03). Vs the 1M uniform-clock model, 7 vs 17: net −10 (p = 0.06). On the 10 eval
  injections both find and miss the same sources (SNR 27 and 37 still missed).
- **Calibration:** in95 0.97–0.99; in68 from conservative (0.87–0.95) towards nominal
  (0.80–0.87). At SNR ≥ 100 rank fell to 0.41: median **+0.021 bins** above the truth (mean
  +0.030 ± 0.005, 6σ).

**Bias vs window position.** Median offset (bins), found sources at SNR ≥ 40, 8 slices of the
f₀ coordinate from −1 to +1 (53–76 sources each):

| model | −1 | | | | 0 | | | +1 |
|---|---|---|---|---|---|---|---|---|
| 1M | −0.018 | −0.032 | −0.130 | −0.146 | −0.045 | −0.033 | +0.009 | −0.026 |
| XS-late | +0.018 | −0.021 | −0.027 | −0.018 | −0.015 | +0.005 | +0.043 | +0.045 |
| 768 | −0.015 | −0.017 | +0.039 | +0.081 | +0.063 | +0.003 | +0.001 | −0.009 |

- F15's "no bias in XS-late" was partly cancellation: ~−0.02 in the lower half, +0.04 at the
  top.
- Each run's bias is a smooth error ~1/4 window (6 bins) wide, with its own shape and sign:
  where the constant-lr weights happened to stop, not physics or data.

**Other.** XS-late's 4 zero-width sources sit at |coord| > 0.995, over half their draws on one
f₀ value; the 768 net has none (4 of 840, medians unaffected).
`outputs/lisa-XS-late-768/checkpoints/1000` restores on the CPU: epoch 1000, 170,077,474
parameters, all finite.
