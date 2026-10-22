# Fisher Score Matching for Simulation-Based Forecasting and Inference

Sui, Pandey, Wandelt, 2025, ICML 2025 ML4Astro workshop-style short paper · [arXiv:2507.07833](https://arxiv.org/abs/2507.07833)

**Evidence: C.** Not peer-reviewed; 1 citation in 1.3 yr (Semantic Scholar, 2026-10-19). Its
claims are the authors' alone: a hypothesis, not a result.

## What they do

Learn the Fisher (likelihood) score $s(x,\theta)=\nabla_\theta\log p(x|\theta)$ by plain regression. If the simulator factorises $\theta\to\theta^*\to x$, then $\nabla_\theta\log p(x|\theta)=\mathbb E_{p(\theta^*|x,\theta)}[\nabla_\theta\log p(\theta^*|\theta)]$ (Eq. 2), so MSE regression of the tractable latent score $\nabla_\theta\log p(\theta^*|\theta)$ on inputs $(x,\theta)$ converges to the Fisher score (Eq. 3). For non-decomposable simulators, add an auxiliary $\tilde\theta$ with $\theta=\tilde\theta+w$, $w\sim\mathcal N(0,\sigma^2 I)$; the learned object is the true score convolved with a kernel narrower than $p(\theta|\tilde\theta)$ (Eq. 5). Learned score is prior-independent (unlike a conditional diffusion model whose training set fixes the prior).

## What they find

Only qualitative figures. Linear Gaussian 2-D toy (Σ_n with ρ=0.5, latent Σ=0.4 I, θ~U[-3,3]², 1e5 triplets): learned arrows overlap analytic score. jax-cosmo weak-lensing $C_\ell$ (5 ℓ bins, θ=(Ω_c, σ_8), σ=1e-3 perturbation, 1e5 triplets): direction accurate everywhere, magnitude underestimated by ~40% (within factor 2) at implausible θ far from the fiducial; HMC posterior with learned score ≈ HMC with true score (contour plot only); Fisher matrices from $\mathbb E[ss^\top]$ match on a grid.

## Implementation details

Tiny MLPs (64-64 / 64-128-128, ELU), lr 1e-2/1e-3, 1k/10k epochs. Noise σ for the indecomposable trick must be small (1e-3 in prior units) — the bias is a convolution. Regression target variance is huge when σ is small (target ∝ w/σ²) — authors don't discuss the variance problem, but it is the same issue as denoising score matching at small noise. Failure mode reported: score magnitude wrong where training density of (x,θ) pairs is low.

## Caveats

2-D parameters, 5-D data; no quantitative metrics, no comparison with NPE/NLE, no ablation on σ or training-set size. Workshop paper.
