# Estimating the Optimal Covariance with Imperfect Mean in Diffusion Probabilistic Models

Bao, Li, Sun, Zhu & Zhang, 2022, ICML 2022 · [arXiv:2206.07309](https://arxiv.org/abs/2206.07309)

**Evidence: B.** Peer-reviewed (ICML 2022; the arXiv page itself lists no venue); 22 citations in 4.4 yr
(OpenAlex, 2026-10-25), below the 30/yr bar for A.

## What they do

Reverse-process variance of a diffusion model is chosen analytically from the trained noise predictor, with a diagonal, state-dependent covariance Σₙ(xₙ) = diag(σₙ(xₙ)²). Full covariances are derived (App. B) but not implemented because sampling from them is slow.
- **Optimum (Thm 3.1):** σₙ*² = λₙ²·1 + γₙ²(β̄ₙ/ᾱₙ)(E[ϵₙ²] − E[ϵₙ]²).
- **Imperfect-mean correction (Thm 3.3):** for a mean built from a noise predictor ϵ̂, the optimum is σ̃ₙ*² = λₙ²·1 + γₙ²(β̄ₙ/ᾱₙ)·E[(ϵₙ − ϵ̂ₙ)²]. The variance absorbs the mean's own error ("conditioned moment matching", Lemma A.1).
- **Two stages (Thm 3.2):** the optimal mean does not depend on the covariance, so the mean is trained first and the covariance second. The paper argues this is at least as good as joint maximum likelihood (Eq. 11).
- **NPR-DPM:** a network gₙ regresses the residual (ϵₙ − ϵ̂ₙ)² with MSE (Eq. 15). **SN-DPM:** a network hₙ regresses ϵₙ² and the estimate subtracts ϵ̂ₙ² (Eq. 10).
- **Relation to Analytic-DPM (ICLR 2022):** that paper gives an isotropic closed-form variance and assumes the mean is optimal; this one is diagonal, corrects for the mean error, and learns the expectations with MSE heads. Analytic-DPM is the A-DDPM/A-DDIM baseline.

## What they find

- Table 2 (FID, even trajectory; A-DDPM / NPR-DDPM / SN-DDPM): CIFAR10 LS, K=10: 34.26 / 32.35 / 24.06; K=1000: 4.03 / 4.10 / 3.65. CIFAR10 CS, K=10: 22.94 / 19.94 / 16.33. CelebA 64, K=10: 28.99 / 28.37 / 20.60. ImageNet 64, K=25: 32.56 / 28.27 / 27.58.
- Table 3 (fewest steps to reach FID ≈ 6, Analytic-DPM / NPR / SN): CIFAR10 25 / 23 / 17; CelebA 55 / 50 / 22; LSUN Bedroom 100 / 90 / 92. Extra cost is ×1.002 to ×1.1.
- Table 4 (NLL upper bound, bits/dim, A-DDPM vs NPR-DDPM): CIFAR10 LS K=10 5.47 vs 5.40; K=1000 3.59 vs 3.57; ImageNet 64 K=25 4.78 vs 4.66. At full steps NPR-DDPM is 3.60 vs 3.57 for Improved DDPM (Sec. 6.2). SN-DPM is poor on likelihood.

## Implementation details

- Shared UNet with two small heads (conv layers): NN₁ outputs ϵ̂ₙ and NN₂ outputs hₙ or gₙ (Eq. 17). Extra compute is at most 10% (4.5% on ImageNet 64).
- Stage two trains only the small head on a pretrained, frozen noise predictor (Alg. 1-2), so no gradient reaches the mean network. Noise predictors come from prior work.
- Sample x₀, n ~ U{1..N}, ϵₙ ~ N(0, I), then take MSE steps "until converged". Optimiser details are in appendices not read.
- Datasets: CIFAR10 (LS, CS, VP-SDE), CelebA 64, ImageNet 64, LSUN Bedroom.

## Caveats

- The variance is for the reverse-process Gaussian step on ϵ-space (noise prediction), not a posterior width in data space. It feeds the sampler directly, as the transition variance.
- Gains are largest at few steps and shrink at full steps (CIFAR10 LS K=1000: 4.03 vs 4.10 for NPR).
- NPR is not uniformly better than SN on FID. SN is worse on likelihood from error amplification in ϵ̂².
- Appendices (E, F) were not read, so the training hyperparameters and the error-amplification derivation are unchecked. Only the first 100k of 197k characters were read.
- No conditional or multi-scale-posterior tasks, and no x-prediction variant.
