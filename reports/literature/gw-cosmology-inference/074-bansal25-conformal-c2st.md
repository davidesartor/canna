# Conformal C2ST: Turning weak classifiers into strong two-sample tests

Bansal et al., 2025/2026, arXiv preprint · [arXiv:2507.17026](https://arxiv.org/abs/2507.17026)

**Evidence: C.** Not peer-reviewed; 1 citation in 1.2 yr (Semantic Scholar, 2026-10-19). Its
claims are the authors' alone: a hypothesis, not a result.

## What they do

Train any classifier to separate joint samples (θ,y) ~ p (prior × simulator) from (θ,y) ~ q (y from the simulator, θ from the NPE). Then rank each q-point's score among m calibration scores from p to get a conformal p-value U. Two variants:
- Uniform: a fresh calibration set per test point, giving exact finite-sample validity.
- Multiple: one shared calibration set, at the same budget as C2ST, with a normalised U-statistic (Hu & Lei 2024) that is asymptotically N(0,1) under the null.
Validity holds for any score; power depends on ranking quality (AUC), and the excess p-value bias is O(ε^{2/3}) in the density-ratio L2 error.

## What they find

- Gaussian perturbations (NPTBench: mean shift, covariance scaling, anisotropic, heavy tails, extra mode, mode collapse), N = 1000, α = 0.05, 200 trials. Conformal variants have higher power than C2ST and match or beat SBC(200), TARP(200), DC and MMDFuse. DC fails Type-I control.
- Lensing (θ ∈ R^256, score-diffusion posterior, bias interpolated by γ), power %:
  - Detection (β = 0): at γ = 0.6, CM 50 and CU(50) 66 vs C2ST 28, SBC 6, TARP 6. At γ = 0.7: 92 / 98 vs 76. Even at γ = 1, SBC is 6 and TARP 24.
  - Classifier degradation at γ = 1: at β = 0.8, CU(50) 86 and CM 74 vs C2ST 40. At β = 0.95, CM 16 vs C2ST 6.
- CIFAR-10 corruptions: highest power throughout.
- App. C, a 3-D sphere manifold embedded in R^100 via RealNVP: conformal tests beat the baselines.

## Implementation details

- Classifier: 3-layer residual MLP, width 256, Adam with cosine schedule, 2000 epochs at LR 1e-5, trained on 1000+1000 pairs.
- Use continuous log-odds as the score, never hard labels. Avoid saturated or clipped outputs that cause ties.
- m ≈ 10–50 captures most of the power gain (m = 50 default). With very weak classifiers, Multiple is more stable than Uniform.

## Caveats

- A global test: it does not localise which parameter or which x is wrong.
- It only tests faithfulness to the simulator's posterior.
- Uniform needs extra p-draws, which are cheap in NPE.
- The experiments are mostly synthetic perturbations.
