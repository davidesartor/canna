# Increasing the Precision of Surrogate Models for Weak Lensing Mass Maps with Flow Matching

Li & Kacprzak, 2026, arXiv preprint · [arXiv:2605.23114](https://arxiv.org/abs/2605.23114)

**Evidence: C.** Not peer-reviewed; 0 citations in 0.4 yr (Semantic Scholar, 2026-10-19). Its
claims are the authors' alone: a hypothesis, not a result.

## What they do

Residual label-conditional flow matching (RLCFM) for 128² κ maps given (Ω_m, σ8). It makes two changes to plain conditional FM:
- (i) **A label-specific Gaussian source** N(μ0(y), σ0(y)²), with σ0 = α · RMS_std(x | y), α = 1.2, and the endpoint width σ1 = β σ0, β = 0.04. The noise width interpolates linearly.
- (ii) **Training in residual space**, r = x − μ0(y), so the network never models the label-dependent mean shift.
- μ0(y) and σ0(y) come from a separate regressor: GP or polynomial ridge, plus a clipped GBDT correction.

## What they find

- Across the grid (80th percentile of 57 cosmologies): W1(pixel) < 0.01, f_Cℓ < 1%, f_Bℓ < 5%, Minkowski functionals < 1–3%, correlation-matrix f_R < 3%.
- Clearly beats the GAN baseline: GAN Cℓ errors are a few % to > 5%, and its correlation-matrix errors 5–15% (up to 40%).
- Converges in about 15 epochs vs 40 for the GAN; 7 h on one A100.
- **Ablation** (11 held-out cosmologies):
  - **A label-specific prior alone (LCFM) is the WORST** variant: highest mean, maximum and standard deviation on every metric, worse than plain CFM.
  - Residual space alone (RCFM) improves the non-Gaussian statistics.
  - Combining both is best.
  - Proposed reason: when the source differs by label, the shared network must represent a vector field whose dependence on y is less smooth, because the sources of neighbouring labels do not overlap. This hurts interpolation.

## Implementation details

- Log-transform, then global standardisation.
- U-Net with conditional ResBlocks and attention at 16².
- Label embedding:
  - Fourier features with a **learnable bandwidth γ, kept small by weight decay** (to avoid high-frequency oscillation in the label direction);
  - hand-crafted physics features (S8 with learnable gating, logs, ratios).
- AdamW, lr 2e-4 cosine-decayed to 3e-5, batch 128 **mixed across cosmologies**, EMA weights for sampling.
- Sampling with a dopri5 ODE solver.
- α > 1 (an over-wide source) helps the large-scale structure.

## Caveats

- An emulator only; there is no posterior or calibration test.
- A 2-parameter grid of 46 training cosmologies.
- The ablation is shown only as curves (Fig. 11), with no table.
- The μ0/σ0 regressor is a hand-tuned side model.
