# Conditional Flow Matching for Bayesian Posterior Inference

Zhai, Jeong, Ročková, 2025; arXiv stat.ML · [arXiv:2510.09534](https://arxiv.org/abs/2510.09534)

**Evidence: C.** Not peer-reviewed; 3 citations in 1.0 yr (Semantic Scholar, 2026-10-19). Its
claims are the authors' alone: a hypothesis, not a result.

## What they do

learn a block-triangular velocity on joint (y,θ) from a joint Gaussian source (dy/dt=f_t(y), dθ/dt=g_t(y,θ)); posterior sampling = integrate θ-part with y* fixed. Optional g_t=∇_θψ_t with ψ convex in θ (ICNN) → conditional Brenier map → credible sets G(y*, ball of radius τ) = Monge–Kantorovich depth regions; consistency: W2 error ≤ 2L e^{2L} ε_N² and asymptotic coverage.

## What they find

sbibm C2ST (lower-to-0.5 better): Gaussian linear 0.71 (Kerrigan 0.89, Wildberger FMPE 0.97); GMM 0.82 (FMPE 0.57 best); Bernoulli GLM 0.92 (FMPE 0.61); two moons 0.74 (R-ABC 0.64); SLCP 0.93 (all ≥0.91). I.e. not competitive with standard FMPE on 2 of 4 shared tasks. Coverage "tracks nominal" — figure only. Training 500–1000 s per task.

## Implementation details

**standardising y and θ to unit variance was essential** — without it the model failed to learn even the prior (App. E.2; SIR variances differ 79:1); ICNN can only represent expansive maps, so use a source with smaller variance than the target.

## Caveats

low-dim sbibm only; no numeric coverage; ICNN ablation qualitative; authors note degradation with high-dim y.
