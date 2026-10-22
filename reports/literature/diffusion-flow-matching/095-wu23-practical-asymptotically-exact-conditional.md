# Practical and Asymptotically Exact Conditional Sampling in Diffusion Models

Wu, Trippe, Naesseth, Blei, Cunningham, 2023, NeurIPS 2023 · [arXiv:2306.17775](https://arxiv.org/abs/2306.17775)

**Evidence: A.** Peer-reviewed (Neural Information Processing Systems); 283 citations in 3.3 yr
(Semantic Scholar, 2026-10-19).

## What they do

Twisted Diffusion Sampler: sample $p_\theta(x^0|y)\propto p_\theta(x^0)p(y|x^0)$ from an *unconditional* diffusion model by SMC over the reverse chain. Optimal twist $p_\theta(y|x^t)$ is intractable; approximate it by the likelihood at the denoiser's prediction, $\tilde p(y|x^t)=p(y|\hat x_\theta(x^t))$, use it both in the proposal (score + $\nabla\log\tilde p$, i.e. gradient guidance) and in the incremental weight $w_t = p_\theta(x^t|x^{t+1})\tilde p(y|x^t)/[\tilde p(y|x^{t+1})\tilde p(x^t|x^{t+1},y)]$, resampling every step. Since $\tilde p(y|x^0)$ is exact, the final target is exact as K→∞ regardless of twist quality. One particle = gradient guidance (DPS-like). Extends to Riemannian diffusion (tangent-normal twists; FrameDiff on SE(3)^N) and to likelihoods with discrete "degrees of freedom" (average twists over masks/rotations; costs one denoiser call total).

## What they find

2-D Gaussian with exact score: mean-estimate MSE falls at the O(1/K) rate for TDS (and IS/TDS-IS where applicable); gradient guidance has a fixed bias that does not shrink with K. MNIST class-conditional: K=64 gives near-perfect classifier accuracy; TDS ≈ TDS-IS > IS, guidance; IS/TDS-IS samples mostly identical due to weight collapse; ESS trace (K=64): TDS ESS rises toward K but halves in the final steps, TDS-IS ≈1, IS ≈6. Inpainting: TDS ESS drops to ≈1 in the final ~20 steps. Protein motif-scaffolding (24 problems, K=8, twist scale 2, 100 rotations × 1000 placements): TDS+FrameDiff beats RFdiffusion on 11/24, loses 11/24 (wins on short scaffolds <100 res 9 vs 3); 0% success without the degrees of freedom.

## Implementation details

systematic resampling every step (ESS-threshold resampling optional); proposal variance = model's; VP forward with T=100 and quadratic variance schedule in sims; final output = denoiser mean. Twist scale γ (exponentiate twist, ≈ guidance scale): γ=2–3 helps at small K; γ=10 distorts; with K=64 any γ>1 lowered human-rated accuracy (target is biased away from the true conditional). "TDS-truncate" variant avoids the final-step ESS collapse. Cost: backprop through denoiser each step × K particles.

## Caveats

asymptotic exactness is w.r.t. the *learned* prior $p_\theta$, not the true one; weight degeneracy late in the chain on sharp likelihoods; requires a stochastic sampler (reverse-SDE transition densities) — not directly a deterministic ODE/flow; no SBI experiments.
