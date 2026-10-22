# The Hidden Linear Structure in Score-Based Models and its Application

Wang et al., 2023, arXiv preprint; workshop-length, no venue stated in v1 · [arXiv:2311.10892](https://arxiv.org/abs/2311.10892)

**Evidence: C.** Not peer-reviewed; 12 citations in 2.9 yr (Semantic Scholar, 2026-10-19). Its
claims are the authors' alone: a hypothesis, not a result.

## What they do

- For a Gaussian data distribution N(μ, Σ = UΛUᵀ), the score at noise σ is linear in x: (1/σ²)(I − U diag[λ/(λ+σ²)] Uᵀ)(μ − x). The probability-flow ODE then has a closed form: the mean stays fixed, the off-manifold part shrinks ∝ σ_t/σ_T, and each PC coordinate scales by ψ = √((σ_t²+λ_k)/(σ_T²+λ_k)).
- Claim: in trained image diffusion models, the **high-noise** score is dominated by this Gaussian (data mean + covariance) score. App. 9.6 argues that any bounded point cloud is indistinguishable from its moment-matched Gaussian at large σ.

## What they find

- EDM models on CIFAR10, FFHQ64 and AFHQv2-64: for σ > 1.08 the Gaussian score explains >99% of the neural score's variance. It beats the isotropic score at all σ, and a 10-class GMM is slightly better again.
- At low noise the neural score departs from the "exact" (point-cloud, memorising) score and looks more like the smoother Gaussian/GMM score.
- Trajectories match the Gaussian solution (MSE < 0.01) for about 9 Heun steps on CIFAR10 (to σ = 1.92) and about 17 on FFHQ/AFHQ (to σ = 4.37).
- "Teleportation": jump analytically to σ_skip, then run Heun.
  - CIFAR10: FID 1.958 → 1.932 with 4–5 of 35 steps skipped (~29% fewer NFE). 10 skips give FID 23.9.
  - AFHQ: 25–30% of NFE saved at equal FID, or FID 2.043 → 2.003 at 6 skips.
  - FFHQ: 10–15% saved with ~3% worse FID (2.464 → 2.561 at 6 skips).
- VP/DDIM: about 40% of the initial steps are skippable on MNIST and CIFAR10. Higher-resolution CelebA-HQ needs more care.

## Implementation details

- (i) A linear bypass: hard-wire the Gaussian score and let the net learn only the nonlinear residual.
- (ii) Whiten the target distribution so the score is more isotropic and faster to learn. The authors offer this as an explanation for latent-diffusion efficiency (a hypothesis, untested).

## Caveats

- Short paper. The bypass and whitening suggestions are untested.
- The result is about **high noise / early sampling**, where the distribution is smoothed far beyond its own scale, and about the data's global mean and covariance. It says nothing about a "late-s Fisher approximation".
- It does not imply that a Fisher-Gaussian base is near-optimal: the linear regime is the high-noise end of the path, a Fisher matrix (local curvature at the truth) is not the moment-matched Gaussian the paper uses, and an FM path from a prior draw is not a Gaussian noising process, so the closed form does not transfer directly.
