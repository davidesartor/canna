# Likelihood-Free Inference with Generative Neural Networks via Scoring Rule Minimization

Pacchiardi & Dutta, 2022, arXiv preprint · [arXiv:2205.15784](https://arxiv.org/abs/2205.15784)

**Evidence: B.** Preprint (arXiv); 22 citations in 4.4 yr (Semantic Scholar,
2026-10-22).

## What they do

The paper trains an amortised posterior Q_φ(·|y) that can only be sampled (a generative network), by minimising a proper scoring rule over simulated pairs (θ_i, y_i), with θ_i ~ prior and y_i ~ P(·|θ_i). The key identity (Eq. 3) is

E_{Y} E_{θ∼Π(·|Y)} S(Q_φ(·|Y), θ) = E_{θ∼Π} E_{Y∼P(·|θ)} S(Q_φ(·|Y), θ).

For a strictly proper S, this is minimised only by Q_φ(·|y) = π(·|y) for every y with p(y) > 0. The mean score over held-out simulated pairs is therefore a samples-only measure of posterior quality that needs no reference posterior.

The scores use the inference convention, which is twice that of Gneiting & Raftery (2007):
- **Energy score**, S_E(P, x) = 2 E‖X − x‖^β − E‖X − X′‖^β, β ∈ (0, 2), with β = 1 throughout.
- **Kernel score**, S_k(P, x) = E k(X, X′) − 2 E k(X, x), with a Gaussian kernel of bandwidth γ. It is strictly proper when the kernel makes the MMD a metric.
- **Patched scores** for gridded θ: w₁S(P, x) + w₂ Σ_p S(P|_p, x|_p) over local patches. The full-vector term keeps the sum strictly proper.

## What they find

- **SLCP** (θ ∈ ℝ⁵, n_train = 10⁵, m = 20; Table 1), as C2ST / calibration error. GAN 0.92 / 0.06, energy 0.95 / 0.07, kernel 0.98 / 0.13. The scoring-rule runs train in 1645 s and 1210 s on one CPU, against 30 963 s for the GAN on a GPU.
- **Two Moons** (Table 2): GAN 0.82 / 0.07, energy 0.73 / 0.03, kernel 0.92 / 0.03. The R² is about 0.5 for all three.
- **Shallow water** (θ ∈ ℝ¹⁰⁰, y ∈ ℝ²⁰ᵏ; Table 3), with the energy score on patches of size 20 and step 10. Energy: NRMSE 0.05, calibration error 0.03, R² 0.89. GAN: 0.07, 0.12 and 0.78. The SBC rank CDFs (Fig. 3) are also closer to uniform for the energy score.
- **Noisy camera** (EMNIST deblurring, θ ∈ ℝ^{28×28}, implicit prior; Table 4). The calibration error is 0.36 for both scoring rules against 0.50 for the GAN, "still quite poor in absolute terms".
- Small m (around 10) is enough in practice. The adversarial method can collapse toward a point mass and underestimate uncertainty. Scoring-rule training avoids the critic and the min-max game.

## Implementation details

- **Unbiased (U-statistic) estimators** from m draws x̃_j ~ P (App. B.1):
  Ŝ_E = (2/m) Σ_j ‖x̃_j − x‖^β − (1/(m(m − 1))) Σ_{j≠k} ‖x̃_j − x̃_k‖^β,
  Ŝ_k = (1/(m(m − 1))) Σ_{j≠k} k(x̃_j, x̃_k) − (2/m) Σ_j k(x̃_j, x).
  They need m ≥ 2 and cost O(m²·d) per pair. Being unbiased for every m, they make mean scores comparable across sample sizes. The plug-in V-statistic is not.
- **Calibration error** (App. C.2.1, after Radev et al. 2020). For each component l, α(l) is the fraction of cases where θ_l falls inside the α-credible interval. The error is the median of |α(l) − α| over 100 values of α in (0, 1), averaged over components. It is marginal only.
- **Point-estimate metrics.** NRMSE = RMSE / (max θ − min θ) and R² = 1 − Σ(θ_i − θ̂_i)² / Σ(θ_i − θ̄)², both for the posterior mean θ̂, per component and then averaged.
- **SBC** (Talts et al. 2018) per component, as rank CDFs with 99% bands.
- **What each metric sees.** The calibration error and per-component SBC are passed by Q = prior. The mean energy or kernel score is not, since its population minimiser is the posterior. For Q = prior, the R² of the posterior mean would be ≈ 0 (this follows from the definition; the paper does not show it). These comparisons are on means over test pairs. No formal tests or score-difference tests are reported.
- The companion forecasting paper (Pacchiardi, Adewoyin, Dueben & Dutta, arXiv:2112.08217) uses the same unbiased estimators for prequential scores. It adds the variogram score and its energy/kernel variants as training objectives.

## Caveats

- Unreviewed preprint (v1 read).
- The scoring rules are used for training. They appear only implicitly as validation metrics: the evaluation uses C2ST, calibration error, NRMSE, R² and SBC, not held-out scores.
- The energy score at β = 1 is known to be weakly sensitive to dependence errors (Scheuerer & Hamill 2015). That is not examined here.
- All calibration metrics are marginal per component. There is no joint calibration check.
- Full text read (arXiv v1).
