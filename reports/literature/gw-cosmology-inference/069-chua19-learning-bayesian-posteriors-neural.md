# Learning Bayesian posteriors with neural networks for gravitational-wave inference

Chua & Vallisneri, 2019, PRL 124, 041102 (2020) · [arXiv:1909.05966](https://arxiv.org/abs/1909.05966)

**Evidence: A.** Peer-reviewed (Physical Review Letters); 101 citations in 7.1 yr (Semantic
Scholar, 2026-10-19).

## What they do

- Train an MLP on prior-drawn (θ, D) pairs with loss −Σ ln q(θⱼ|Dⱼ). This is the amortised-NPE cross-entropy, derived as the expected KL from the true posterior, and needs no likelihood evaluations.
- Output forms: a Gaussian, a Gaussian mixture, or a softmax histogram.
- It works on 1-D or 2-D marginals, one network per parameter or pair. **Marginalisation is free:** just drop nuisance parameters from the loss.

## What they find

- **Setup:**
  - Waveform: TaylorF2 2.5PN through the ROMAN reduced-basis net (d ≈ 200 coefficients).
  - Parameters (M_c, η, χ₁, χ₂, ρ), masses 1.25–10×10⁵ M☉ (LISA-like MBH).
- **(M_c, η) network:**
  - 8×1024 leaky-ReLU MLP producing one bivariate Gaussian.
  - A fresh 10⁵ signals every epoch, ~10⁹ examples to convergence.
  - Means good to ~98% of the prior width, but **σ overestimated by up to 3×**.
- **Accuracy degrades:**
  - at small M_c, where there is more information;
  - near prior boundaries (truncated posteriors).
- **Broader priors failed to converge.** "Acceptable convergence only when the priors are confined to smaller subspaces", because Fisher-narrow posteriors in a large prior are "difficult to resolve".
- At ρ = 2, the histogram output captured multimodal M_c posteriors.

## Implementation details

- **Data is the reduced-basis projection β** with unit-variance white noise, β = α(θ) + ν, ν ~ N(0, I): compressed and whitened, so noise is drawn trivially.
- Templates are normalised to unit SNR, and amplitude enters as ρ·α, so ρ is an explicit parameter.
- "Effectively infinite" training sets from batched GPU waveforms remove overfitting.
- **Ground truth:** brute-force oversampling of the reduced likelihood.
- **Metrics:** log covariance-volume accuracy and principal-axis angle accuracy.

## Caveats

- Low-dimensional marginals only.
- Gaussian or mixture outputs are too rigid; MoG had trouble with multimodality.
- Toy waveform, no detector response, no extrinsics.
