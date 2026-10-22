# CP4SBI: Local Conformal Calibration of Credible Sets in Simulation-Based Inference

Cabezas et al., 2025, arXiv preprint, v3 Jun 2026 · [arXiv:2508.17077](https://arxiv.org/abs/2508.17077)

**Evidence: B.** Peer-reviewed (Philosophical transactions. Series A, Mathematical, physical,
and engineering sciences); 9 citations in 1.2 yr (Semantic Scholar, 2026-10-19).

## What they do

Treat any Bayesian credible-set score s(θ, x) as a conformal nonconformity score and calibrate its threshold on a held-out set of (θ, x) pairs drawn from the joint. There are two ways to make the threshold local:
1. **LoCart.** A regression tree predicts s from x. Its leaves partition x-space, and the threshold is the (1+1/n_j)(1−α) quantile within the leaf. The tree's features are augmented with an estimate of Var[s | x].
2. **CDF.** Transform the score to s′ = F̂(s(θ, x) | x), the empirical CDF of the score under M = 1000 draws from p̂(·|x). Then take one global quantile of s′. This is PIT-style: if p̂ = p, s′|x ~ U(0,1).

## What they find

- Setup:
  - 10 sbibm tasks at 1−α = 0.9.
  - Budget of 10,000 sims, split 80% training and 20% calibration (B = 2000).
  - Estimators are NPE (sbi flows) and NPSE (diffusion).
  - Evaluated with conditional-coverage MAE over 500 or 10 reference x's, and with marginal coverage on 2000 test pairs.
- NPE: both variants are significantly best on conditional MAE in 8/10 tasks. Lotka–Volterra is the exception, where global CP wins.
- NPSE: LoCart is among the best in 7/10. HDR recalibration (Chung+24) never reaches marginal coverage.
- Budget dependence:
  - At a 2,000-sim budget, CDF is best in 9/10, and LoCart is no better than global CP because the leaves are too sparse.
  - At 20,000, both variants win 7/10.
- Region sizes (Table 2): self-calibration is too small on Gaussian Mixture (8.1 vs about 9.9) and too large on Two Moons. The conformal variants adapt.
- The neural mass model (3 params, B = 4000) is qualitative only.

## Implementation details

- **Sample-only models (flow matching, diffusion):** use a KDE HPD score s = −Σ_l K(θ, θ_l) with Scott's-rule bandwidth over L posterior draws per x. Alternatives:
  - a standardised distance |θ − mean| / std;
  - an empirical-quantile score, max(q̂_α1 − θ, θ − q̂_α2), for 1-D intervals.
- **Calibration data:** hold out about 20% of the training sims, or simulate fresh ones. LoCart's tree needs at least 300 samples per leaf at 10–20k (75 at 2k), plus cost-complexity pruning.
- **No split in practice:** they use the whole calibration set for both the tree and the cutoffs. Strict validity would need a split, but they report little empirical impact.
- **Marginal parameters:** for a subset or transform φ = g(θ), train or derive p̂(φ|x) and calibrate on (g(θ_i), x_i). For example, this gives a 2-D region from a 10-D problem.
- **Cost:** the CDF variant needs M = 1000 posterior draws for every calibration x and for x_obs, but no extra simulations.

## Caveats

- Guarantees are with respect to the training prior. Under misspecification or a shifted population they say nothing.
- "Local" means local in x-partition cells, not truly conditional.
- The KDE HPD score degrades with dimension. Every experiment uses at most 10 parameters, and sizes are reported only in 2-D.
- It recalibrates *regions*, not the posterior: you get a set, not corrected samples.
