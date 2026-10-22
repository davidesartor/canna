# F6 (2026-10-01): the loud-source floor is nearly invisible to the flow-matching loss
- Excess loss is E|v_θ − v*|² ≈ s₀·σ_θ per coordinate, whatever the true σ: the loss rewards
  absolute narrowness, not the right width.
- The current floor costs ~1.7×10⁻⁴ of logged loss, below the epoch scatter.

**Model.** One coordinate, 1-D Gaussian: x₀ ~ N(0, s₀²), s₀ = 0.58 (std of U[−1, 1]); true
posterior x₁ ~ N(0, σ²); a model whose field is exactly that of a N(0, σ_θ²) posterior.

**Cost.** σ_θ = 0.18 bins (0.0075 flow units) on the f₀ of ~half the sources (SNR above ~60)
costs **1.7×10⁻⁴** of flow loss (averaged over 4 sources × 11 coordinates), vs 5×10⁻⁴ epoch
scatter and 1.7×10⁻² per doubling of steps. No training at this weighting will push on it;
bigger networks and longer runs lower only the visible coarse structure (detection, sky,
amplitude).

**Why: time sampling.** A σ-wide posterior's structure lives at 1 − t ≲ σ/s₀: 0.014 for the
current floor (1.4% of uniform-t samples), 3×10⁻⁵ for Fisher-width loud sources. Log-uniform 1
− t on [10⁻⁶, 1] weights the floor **29×** more; a 50/50 mix with uniform t **15×**
(`loss_sensitivity.py`).

**Time embedding.** `SinusoidalEmbed` angular frequencies span only 1 to 2π per unit t, so the
network cannot tell 1 − t = 10⁻³ from 10⁻⁴: emphasising late times also needs log(1 − t)
conditioning.
