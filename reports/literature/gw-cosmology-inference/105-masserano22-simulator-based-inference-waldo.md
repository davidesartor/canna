# Simulator-Based Inference with WALDO: Confidence Regions by Leveraging Prediction Algorithms and Posterior Estimators for Inverse Problems

Masserano, Dorigo, Izbicki, Kuusela, Lee, 2022; AISTATS 2023 · [arXiv:2205.15680](https://arxiv.org/abs/2205.15680)

**Evidence: B.** Peer-reviewed (International Conference on Artificial Intelligence and
Statistics); 23 citations in 4.4 yr (Semantic Scholar, 2026-10-19).

## What they do

- Test statistic: τ(D;θ₀) = (E[θ|D] − θ₀)ᵀ V[θ|D]⁻¹ (E[θ|D] − θ₀), using posterior moments from an NPE (Monte Carlo) or from mean and variance regressors.
- Critical values C_{θ,α}: quantile regression of τ on θ over simulations with θ drawn across Θ.
- Confidence set by Neyman inversion over a θ₀ grid: {θ₀ : τ(D;θ₀) ≤ Ĉ_{θ₀,α}}.
- Guarantee: conditional (frequentist) coverage for every θ, whatever the quality of the estimator or the prior, provided the quantile regressor is consistent. The prior and estimator quality affect only power (set size).
- Diagnostic: a GAM fit of 1[θ ∈ R(D)] against θ.

## What they find

- 2-D Gaussian mixture with NPE (sbi, B = 100k, B′ = 30k): 95% NPE credible regions cover as little as about 0–10% for θ far from the prior mean, while WALDO sets are nominal everywhere. An informative prior gives smaller sets than a uniform one.
- Muon calorimeter (51,200-dimensional input, CNN): WALDO reaches 68.3% coverage with all three input types. Naive 1σ prediction intervals over- or under-cover depending on energy, with the largest bias at high energy.
- Quantile regression needs orders of magnitude fewer simulations than per-point Monte Carlo calibration (tested at p = 1 and p = 10).

## Implementation details

Use quantile-regression forests or GBMs for the critical values. Standardizing by the posterior variance gives more power (Pareto example). Hold out separate splits for critical values, final sets and diagnostics.

## Caveats

Neyman inversion on a θ grid is costly as dimension grows (tested up to p = 10). The Wald form gives ellipsoidal-ish sets, poor for multimodal or circular parameters. It is frequentist coverage, not Bayesian calibration.
