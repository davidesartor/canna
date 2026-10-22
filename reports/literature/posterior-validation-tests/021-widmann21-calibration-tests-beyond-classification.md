# Calibration Tests Beyond Classification

Widmann, Lindsten & Zachariah, 2021, ICLR 2021 · [arXiv:2210.13355](https://arxiv.org/abs/2210.13355)

**Evidence: B.** Peer-reviewed (International Conference on Learning Representations); 26
citations in 5.5 yr (Semantic Scholar, 2026-10-22).

## What they do

The paper defines calibration for any probabilistic predictive model, with targets of any dimension, and builds kernel tests for it.

- **Definition.** A model P that maps features X to a predictive distribution P_X is calibrated iff (P_X, Y) and (P_X, Z_X) have the same distribution, where Z_X ~ P_X. In words: among all inputs that get the same prediction, the targets follow that prediction.
- **Calibration error.** CE_F = sup_{f∈F} |E f(P_X, Y) − E f(P_X, Z_X)|. With F the unit ball of an RKHS on (predictions × targets), this is an MMD between the two joint distributions. With a characteristic kernel, it is zero iff the model is calibrated. They call it the squared kernel calibration error, SKCE.
- **Kernel.** A tensor product k((p, y), (p′, y′)) = k_P(p, p′) · k_Y(y, y′). The kernel on predictions is k_P(p, p′) = exp(−λ d_P(p, p′)^ν), valid for any λ > 0 and ν ∈ (0, 2] when d_P is a Hilbertian metric. The 2-Wasserstein distance between Gaussians or Gaussian mixtures is one example.
- **Estimators.** A biased estimator, the unbiased quadratic estimator SKCE_{k,n}, and block estimators SKCE_{k,B} that average unbiased estimates over ⌊n/B⌋ blocks of size B. The block estimator costs O(Bn).
- **Tests.** For fixed B, √⌊n/B⌋ SKCE_{k,B} is asymptotically normal, so the p-value is analytic. For B = n, the null distribution is a weighted sum of centred χ²₁ variables, approximated by bootstrap. A third test, CME, uses an unnormalised mean embedding at a few test locations with a Hotelling statistic.
- **Sampling Z_X.** The term E_{Z∼P_x} f(P_x, Z) can be computed analytically for Gaussian predictions. Otherwise it can be estimated by Monte Carlo from samples of P_x, which keeps the estimators unbiased and consistent.

## What they find

- **OLS example** (Fig. 1). A Gaussian OLS model fitted to 100 points looks calibrated in a quantile plot on 50 validation points. The SKCE test, with 100,000 bootstrap samples, rejects calibration at p < 0.05.
- **Synthetic Gaussians** (d = 1 and d = 10). Predictions N(c·1_d, 0.1² I_d) with c ~ U(0, 1). The uncalibrated case shifts the first target coordinate's mean by 0.1. Over 500 datasets of n ∈ {4, 16, 64, 256, 1024}:
  - All tests hold the level asymptotically.
  - The B = 2 block test is much cheaper but has lower power.
  - B = √n "still achieves good test power" for n > 30.
  - Fixed-B tests are "orders of magnitudes faster" than the bootstrap test (1000 bootstrap samples).
  - The CME test with 10 locations converges much more slowly to the level.
- **Friedman 1 regression** (10-D features, ten Gaussian models with a shallow-network mean and one scalar variance). MSE, NLL, SKCE and the p-value all show the models overfit after 1500 training iterations. SKCE isolates calibration, while NLL mixes calibration and accuracy.

## Implementation details

- **Prediction kernel.** For Gaussian predictions they use exp(−W₂(p, p′)), with W₂ in closed form from the means and standard deviations. For sample-only predictions, some distance between sample sets is needed. This is my reading, not a claim of the paper.
- **Target kernel.** A Gaussian kernel, e.g. exp(−(y − y′)²/2) in the OLS example.
- **Block size.** B = √n is the paper's practical trade-off between cost and power.
- **Code.** github.com/devmotion/Calibration_ICLR2021. Julia packages CalibrationErrors.jl and CalibrationTests.jl.

## Caveats

- Calibration here conditions on the prediction P_X, not on X. A model that outputs the same distribution for every input is calibrated whenever that distribution equals the marginal of Y. So a posterior equal to the prior, with θ drawn from that prior, passes. This is my reading, not a claim of the paper.
- Too-wide predictions break the definition and are detectable in principle. The paper's synthetic power study only uses a mean shift.
- Power depends on k_P. If k_P cannot tell predictions apart, the test reduces to a marginal check.
- The experiments are small (n ≤ 1024, d ≤ 10). There is no simulation-based-inference example.
- Chatterjee et al. 2024 report slightly inflated SKCE type-I error (0.064–0.076 at 5%) in their comparison.
