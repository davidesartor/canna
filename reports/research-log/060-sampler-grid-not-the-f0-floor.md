# F60 (2026-10-24): the sampler's grid is not full-B's f₀ floor; the eval's 32 uniform steps are converged, and steps late in the path widen f₀

- **The eval's sampler is converged:** f₀ widths on the uniform midpoint grid change by less than
  15% from 32 to 256 steps on both runs (full-B 0.46 → 0.40 bins, narrow-S 0.025 → 0.028).
- **More steps late in the path make f₀ wider, in every SNR band:** at 256 steps full-B is
  0.62–0.65 bins on the power-3 and log grids (narrow-S 0.056–0.063), and the CRPS worsens with
  them. The network's x̂₁ late in the path is worse than at 1 − s = 1/64, not better.
- **So the hypothesis behind this run is wrong:** the floor is not the last query sitting at
  1 − s = 1/64 with x_t's f₀ a token off (the frequency-RoPE path error). It is the network's own
  x̂₁.
- **New suspect, untested:** the network writes x̂₁ as an absolute value
  (`x_unembed(h)`) from the bf16 backbone's tokens. In x_f both runs share one floor, 2.6 × 10⁻⁴
  (full-B, 0.4 bins of 1536 per unit) and 4.4 × 10⁻⁴ (narrow-S's loud 0.021 bins of 48), the size
  of a readout of bf16-rounded features (2⁻⁹). fml predicted the velocity and moved draws by
  v·Δs, which scales any output error by the path left (3 × 10⁻⁵ in its last step). This is the
  "floor scales with the window" of [F59](059-oneshot-first-runs-narrow-s-full-b.md).

**Runs.** [sampler-convergence](../posterior-eval/sampler-convergence/README.md) probe on F59's
checkpoints (`lucky-falcon-full-B` epoch 48, `misty-meadow-narrow-S` epoch 65, both at their
training commit 3e5bdd4), TREX A100, jobs 15711973 and 15711974, ~5 min each. 256 draws on the 10
eval injections; uniform, power-2, power-3 and log grids at 4–256 steps, midpoint and Euler. The
probe's grid sampler reproduces the eval's on its own grid exactly (max |Δx| 0).

**f₀ width [bins], median per source SNR band** (sources per band: full-B 4, 11, 6, 7, 12;
narrow-S 8, 5, 7, 9, 11):

| run | sampler | 0–15 | 15–40 | 40–100 | 100–300 | ≥ 300 |
|---|---|---|---|---|---|---|
| full-B | uniform 32 (the eval) | 1.85 | 0.47 | 0.76 | 0.43 | 0.44 |
| full-B | uniform 256 | 1.97 | 0.42 | 0.52 | 0.49 | 0.35 |
| full-B | power-3 256 | 0.90 | 0.73 | 0.71 | 0.73 | 0.50 |
| full-B | log 256 | 1.40 | 0.71 | 0.68 | 0.72 | 0.51 |
| narrow-S | uniform 32 (the eval) | 0.045 | 0.018 | 0.023 | 0.025 | 0.026 |
| narrow-S | uniform 256 | 0.103 | 0.025 | 0.056 | 0.023 | 0.023 |
| narrow-S | power-3 256 | 0.123 | 0.061 | 0.071 | 0.060 | 0.026 |
| narrow-S | log 256 | 0.122 | 0.053 | 0.069 | 0.048 | 0.026 |

**What follows.**
- **Test the bf16 readout without training:** evaluate both runs with an fp32 backbone
  (`LISA_BF16=none`), and run the [precision-error](../architecture/precision-error/README.md)
  probe on them. A floor that drops by several times confirms it; an unchanged one means the
  floor is learned. The runs trained in bf16, so this is a diagnostic, not their performance.
- **If confirmed, keep the bf16 backbone and predict a correction:** x̂₁ = x_t + (1 − s)·head(h),
  with the same x-prediction loss, so the rounding vanishes along the path as it did on fml.
- **Not worth trying:** float64 parameters. The sampler state is float64 already, x_t enters the
  network in float32 and the head runs in float32; the digits are lost inside the backbone.
- F59's other suspects stay: time_power 3 → 1, the learning rates, the one-way data stream.
