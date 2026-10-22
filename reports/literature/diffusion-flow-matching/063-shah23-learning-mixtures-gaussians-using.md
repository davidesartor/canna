# Learning Mixtures of Gaussians Using the DDPM Objective

Shah et al., 2023, NeurIPS 2023 · [arXiv:2307.01178](https://arxiv.org/abs/2307.01178)

**Evidence: B.** Peer-reviewed (Neural Information Processing Systems); 72 citations in 3.3 yr
(Semantic Scholar, 2026-10-19).

## What they do

- First provable result that gradient descent on the DDPM objective learns a distribution family: mixtures of spherical Gaussians.
- The student is architecture-matched to the true score: s_μ(x) = tanh(μᵀx)μ − x for K = 2, i.e. a two-layer tanh net, and a softmax-weighted version for K components.
- The DDPM gradient behaves like two classical algorithms:
  - **High noise** (t = O(log d)): −∇L ≈ (2μ*_t μ*_tᵀ − r I)μ, i.e. power iteration on the second-moment matrix. From random init it converges in angle to μ*.
  - **Low noise** (t = O(1)): one GD step ≈ (1−η)μ + η·(EM M-step) for 2-GMM, and ≈ gradient-EM for K-GMM. This contracts ‖μ − μ*‖ from a warm start.

## What they find

- Thm 7: for ‖μ*‖ > c, a two-stage schedule recovers μ* to ε with poly(d, 1/ε) time and samples: GD at high noise from random init (H = poly(d, 1/ε)), then GD at low noise initialised from stage 1.
- Thm 13: for 1/poly(d) separation, projected GD at high noise only (onto a ball of radius R = √(mean‖x‖² − d)) gets to ε, with poly(d)/ε⁸ complexity.
- Thm 16: for K components with Ω(√log min(K, d)) separation, a **warm start** is required (Assumption 15); low-noise GD then converges.
- The two-stage algorithm has better statistical rates than the projection-only one.

## Implementation details

- (i) A coarse-to-fine curriculum in noise level: train at high noise first to find directions, then low noise to refine.
- (ii) Global symmetric-mixture structure is found by a **second-moment** (power-iteration) signal, because the first moment vanishes by symmetry.
- (iii) Without a warm start, K > 2 is not covered.

## Caveats

- The theory is for an architecture that is already the exact score family, with separate parameters per noise level (one μ_t per t). It is not a time-conditioned general network.
- No experiments.
- The algorithm has two stages, but its curriculum is per-noise-level training with initialisation passed from high to low noise; it is not a time-density schedule for a shared network.
