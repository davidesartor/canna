# Diagnostics for Conditional Density Models and Bayesian Inference Algorithms

Zhao, Dalmasso, Izbicki, Lee, 2021, UAI 2021 · [arXiv:2102.10473](https://arxiv.org/abs/2102.10473)

**Evidence: B.** Peer-reviewed (Conference on Uncertainty in Artificial Intelligence); 37
citations in 5.7 yr (Semantic Scholar, 2026-10-19).

## What they do

- **Theorem 1:** global PIT or coverage checks pass whenever the estimate depends on x only through some g(x), e.g. a model that ignores a relevant covariate. Global calibration is therefore not enough.
- **Local Coverage Test (LCT):** for a grid of levels α, regress W^α = 1[PIT < α] on x. This gives r̂_α(x), the local coverage at x. The statistic is T(x) = mean_α (r̂_α(x) − α)², with a Monte Carlo null from uniform pseudo-PITs.
- **Amortized local P-P plots:** show bias (offset) versus dispersion (S-shape) at any x without new sampling.
- Multivariate θ: use HPD values (Theorem 3: uniform under the local null) or 1-D projections.

## What they find

- Omitted variable: X ~ N(0,Σ) with ρ = 0.8 and Y|X ~ N(X1+X2, 1). Both models pass PIT histograms. The global test rejects the model using X1 only (p = 0.004) and accepts the full one (p = 0.894).
- Galaxy images (400 dimensions): a single-Gaussian ConvMDN fails globally (p < 0.001), and the LCT localizes the failure to the skewed and bimodal subpopulations.
- NPE with K = 7 mixture components (chosen by KL loss over K = 2–10) still fails the global test (p < 0.001), and the LCT pins it to the 10% subpopulation.

## Implementation details

Run the global test first, then the LCT. Pick the regressor (kernel, random forest or NN) with the lowest validation MSE, since test power tracks MSE. Use Benjamini–Hochberg across locations. The cost is repeated regressor refits for the null.

## Caveats

A uniform local HPD value is necessary, not sufficient. The Monte Carlo null is expensive. Examples are low-dimensional in θ.
