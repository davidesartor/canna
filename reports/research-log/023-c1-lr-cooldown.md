# F23 (2026-10-06): C1 — the lr cooldown halves the floor, removes the bias and finds more sources

- The constant lr was the main cause of the floor and of F19's bias: cooling cuts the loud f₀
  floor 2.3× (55× → 25× ideal).
- Detection is the best of any model so far; the warp's detection cost (F15) is gone.
- fp32 on the A100 is correct with autotuning off (F22 holds).

**Runs.** C1 training (job 13780234): epochs 1000 → 1200, lr linear to 0, 3 h 36 min. Scorecard
13793687, eval 13793690, on F15/F19's 840 sources; widths are medians over sources both models
found.

| SNR | n | found, 768 at 1M → cooled | f₀ width (bins) | per-source ratio [16–84%] | × ideal (cooled) | in68 | in95 | rank |
|---|---|---|---|---|---|---|---|---|
| < 15 | 148 | 0.47 → **0.54** | 0.163 → 0.103 | 0.63 [0.50–0.77] | 2 | 0.80 → 0.78 | 0.97 → 0.97 | 0.49 → 0.53 |
| 15–40 | 157 | 0.83 → **0.86** | 0.119 → 0.067 | 0.53 [0.42–0.67] | 3 | 0.87 → 0.90 | 0.99 → 1.00 | 0.48 → 0.50 |
| 40–100 | 151 | 0.94 → **0.97** | 0.091 → 0.040 | 0.46 [0.34–0.57] | 5 | 0.81 → 0.90 | 0.98 → 0.99 | 0.47 → 0.49 |
| ≥ 100 | 384 | 0.995 → 0.995 | **0.087 → 0.037** | **0.44 [0.35–0.57]** | 25 | 0.82 → 0.93 | 0.98 → 0.99 | 0.41 → 0.49 |

- **Loss:** 10-epoch mean 0.4499 → **0.4276** (−0.022); at the constant-lr pace (−0.002 per
  100k) that would take ~1.1M more steps.
- **Floor:** loud median ratio **0.440** (95% 0.427–0.453); 0.038 / 0.039 / 0.035 / 0.036 bins
  at SNR 100–200 / 200–400 / 400–800 / ≥ 800, still flat in SNR. Below the rule's 0.06 bins, so
  optimiser noise dominated.
- **Bias gone:** loud mean offset +0.029 ± 0.005 → **+0.007 ± 0.005** bins, median |offset|
  0.042 → 0.013, window-position profile 0.081 → **0.008**, rank 0.41 → 0.49. Passes the rule
  (≲ 0.03 bins, rank 0.45–0.55).
- **Detection:** 21 sources found only by the cooled model vs 1 only by the 1M one (33 vs 2
  against XS-late). Beats the uniform-clock 1M model (F15) in every band: 0.54 / 0.86 / 0.97 vs
  0.52 / 0.85 / 0.95. The planned `time_power 2` step is no longer needed.
- **Calibration** conservative everywhere (in68 0.78–0.93, in95 0.97–1.00); in68 = 0.93 at SNR
  ≥ 100 means posteriors are still too wide.
- **Corner plots** (`outputs/lisa-XS-late-768-cool/corner`, visual only): fewer stray contours;
  the intermediate-amplitude f₀–A blob is gone.

**A100 fp32 at autotune level 0** (job 13793694, 21 min vs 3 min at level 3) reproduces the CPU
fp32 scorecard (T3'): widths/offsets to a median 4×10⁻⁷ bins (max 6×10⁻⁵), identical found sets
and in68/in95 flags, ranks within one draw in 256. Autotuning at levels 3 and 4 picks wrong
fp32 kernels. fp32 on the GPU is usable at ~7× the eval time; bf16 needs none of this.
