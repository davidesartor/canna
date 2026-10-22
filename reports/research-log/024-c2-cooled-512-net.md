# F24 (2026-10-06): C2 — once cooled, the 512 net matches the 768's floor but finds fewer sources; S runs at 768

- The cooldown helps the 512 net exactly as much as the 768: F23's gain is the schedule's.
- The loud floor (25× ideal) does not depend on width, so it is not capacity.
- Width buys detection and intermediate-SNR precision. **S runs at 768.**

**Runs.** C2 (job 13849421): XS-late (512 × 8), epochs 1000 → 1200 with C1's cooldown and
resumed optimizer state, 2 h 11 min. Scorecard 13867994 on the same 840 sources.

| SNR | n | found: 512 at 1M / 512 cooled / 768 cooled | f₀ width, 512c / 768c (bins) | ratio 512c / 768c [16–84%] | ratio 512c / 512 at 1M | in68, 512c / 768c |
|---|---|---|---|---|---|---|
| < 15 | 148 | 0.45 / 0.50 / **0.54** | 0.118 / 0.102 | 1.12 [0.96–1.52] | 0.70 | 0.76 / 0.78 |
| 15–40 | 157 | 0.81 / 0.85 / **0.86** | 0.077 / 0.068 | 1.15 [0.96–1.45] | 0.59 | 0.92 / 0.90 |
| 40–100 | 151 | 0.92 / 0.95 / **0.97** | 0.044 / 0.041 | 1.13 [0.91–1.43] | 0.51 | 0.93 / 0.90 |
| ≥ 100 | 384 | 0.99 / 0.995 / 0.995 | 0.039 / 0.037 | **1.03 [0.85–1.32]** | 0.44 | 0.93 / 0.93 |

- **Loss:** 0.4557 → **0.4352** (−0.020, vs −0.022 for the 768); the ~0.008 gap between widths
  is unchanged.
- **Cooldown:** loud width ×0.443 against its own 1M checkpoint (C1: ×0.440). Window-position
  profile 0.045 → **0.009** bins, loud offset +0.011 ± 0.005. Calibration as the 768's (in95
  0.97–1.00 for both).
- **Floor:** at SNR ≥ 100 the 512 is 1.033× the 768 (95% 1.012–1.053); 0.044 / 0.040 / 0.036 /
  0.035 vs 0.038 / 0.039 / 0.035 / 0.036 bins by SNR.
- **Below SNR 100** the 768 is 11–15% narrower and finds 18 sources the 512 misses vs 7 the
  other way, net 11 of 840 (sign test p = 0.04; +4.1, +1.3, +2.0 points by band) — the same +11
  as at constant lr (F19).

**Decision** (rule set before the run): loud width within 1.1× (yes, 1.03), found fractions
within 1 point in every band (no). So **S runs at 768**: `configs/S-late-768.yaml`, ~34 h in
two chained 24 h slots.
