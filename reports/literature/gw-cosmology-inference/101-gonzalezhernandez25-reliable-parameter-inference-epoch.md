# Reliable Parameter Inference for the Epoch of Reionization using Balanced Neural Ratio Estimation

González-Hernández, Wolfson, Hennawi, 2025, short workshop-style paper; venue not stated in the HTML · [arXiv:2511.02808](https://arxiv.org/abs/2511.02808)

**Evidence: C.** Not peer-reviewed; 1 citation in 1.0 yr (Semantic Scholar, 2026-10-19). Its
claims are the authors' alone: a hypothesis, not a result.

## What they do

- 2-parameter model (λ_mfp, ⟨F⟩).
- Baseline: a Gaussian likelihood with a per-grid-point mean and covariance, nearest-grid-point interpolation over 557 grid points, sampled with emcee.
- BNRE: the default sbi ResNet classifier (2 blocks × 50 units) on concatenated (ξ, θ), with balance weight γ = 100, trained on 557 × 500 mocks. The mocks are a subset of those already made for the covariances, so the training data cost nothing extra.
- Posterior sampling with NUTS (pyro) through the differentiable ratio.

## What they find

- Gaussian likelihood on 500 held-out mocks:
  - TARP curve clearly off the diagonal.
  - SBC skewed for λ_mfp (bias) and U-shaped for ⟨F⟩ (over-confident).
- BNRE: TARP near the diagonal within the 100-bootstrap 16–84% bands; SBC inside the uniform band.
- γ ablation (10, 100, 1000): all near-calibrated. ⟨F⟩ ranks are slightly U-shaped at γ = 10 and slightly ∩-shaped at γ = 1000, so γ trades sharpness for conservativeness as expected.
- Default sbi NPE (MAF, 5 transforms × 50 units, tanh, early stopping): strongly U-shaped SBC, i.e. over-confident.
- Cost: 7.9 CPU-h for NPE vs 1.2 CPU-h for BNRE.

## Implementation details

- Reuse the covariance mocks as an SBI training set.
- Bootstrap the TARP curve over test cases to get error bands.
- Report SBC and TARP together: TARP measures global coverage, SBC catches per-parameter bias versus over-confidence.

## Caveats

- Only 2 parameters on a 557-point grid, so the ratio estimator sees θ only at grid nodes.
- No hyperparameter tuning for NPE, so "NPE over-confident" is a default-settings result (the authors say so).
- No tight-posterior regime is tested.
- No numbers beyond figures.
