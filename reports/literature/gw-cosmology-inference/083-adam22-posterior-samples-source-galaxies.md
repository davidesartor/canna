# Posterior samples of source galaxies in strong gravitational lenses with score-based priors

Adam et al., 2022, NeurIPS 2022 ML4PS workshop · [arXiv:2211.03812](https://arxiv.org/abs/2211.03812)

**Evidence: B.** Not peer-reviewed; 35 citations in 3.9 yr (Semantic Scholar, 2026-10-19).

## What they do

Learn a pixel prior over source galaxies with a VE-SDE score model, then sample the posterior with a reverse SDE.
- The posterior score is the prior score s_θ(x_t, t) plus the gradient of a convolved likelihood N(y | A x_t, (σ_N² + σ(t)²) I).
- The lensing operator A (known mass model) makes the problem linear-Gaussian.

## What they find

- Reconstructions fit the observation almost down to the noise level (residual χ² shown).
- The posterior standard deviation from 320 samples is smaller near the caustics (high magnification), as expected.
- Out-of-distribution test (a "7"-shaped galaxy from DALL-E): reconstruction is excellent at low noise and drifts to prior-like galaxies as noise increases. This is a qualitative prior–likelihood balance check.
- No coverage or calibration statistics are reported.

## Implementation details

- VE SDE with σ(t) = σ_min (σ_max/σ_min)^t.
  - σ_max = 263.4: the largest pairwise Euclidean distance in the training set (Song & Ermon technique 1).
  - σ_min = 1e-4: about the scale of the finest image detail.
- N = 8000 Euler-Maruyama steps, from Song & Ermon technique 2. A step ratio γ = (σ_max/σ_min)^{1/N} is chosen so a sample at one level is within the 3σ shell of the next with probability p(N) ≳ 0.5. For n = 256², N = 2000 gives p = 0.64 and N = 8000 gives p = 0.99.
- NCSN++ (score_sde_pytorch), trained on 4 V100s, batch 16, about 350k steps (about 70 h), on 2,059 PROBES galaxies at 256².
- Appendix A derives the general convolved likelihood p_t(y | x_t) ≈ N(y | A x_t, Σ_y + A Σ_t Aᵀ). It is exact as t → 0 and valid when the prior is broad compared with the likelihood: a Neumann expansion of Σ_c around Σ_t, controlled by Σ_t Σ_x0⁻¹. It may break near t ≈ 1, where σ_max² is comparable to the largest prior eigenvalue.
- They further set A Aᵀ ≈ I so the covariance is diagonal and needs no matrix inversion.

## Caveats

- The approximation is uncontrolled at early (noisy) times.
- The mass model κ is assumed known.
- 8000 network evaluations per sample.
- Only qualitative validation.
