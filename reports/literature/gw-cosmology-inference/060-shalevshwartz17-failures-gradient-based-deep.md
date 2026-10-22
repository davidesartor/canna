# Failures of Gradient-Based Deep Learning

Shalev-Shwartz et al., 2017, ICML 2017 · [arXiv:1703.07950](https://arxiv.org/abs/1703.07950)

**Evidence: A.** Peer-reviewed (International Conference on Machine Learning); 235 citations in
9.6 yr (Semantic Scholar, 2026-10-19).

## What they do

four families of simple problems where end-to-end gradient training fails, each with a mechanism.
- **§2, uninformative gradients.** For a hypothesis class {F_h}, Theorem 1 bounds the variance of the loss gradient across target choices h: Var_h(∇F_h(w)) ≤ G²/|H|. The bound is exponentially small whenever the targets are pairwise near-orthogonal, so the gradient carries almost no information about which target is being learned.
  - Parities: |H| = 2^d.
  - Theorem 2 (from Shamir 2016) covers linear-periodic targets ψ(vᵀx) with ‖v‖ = r: Var ≤ exp(−Ω(d)) + exp(−Ω(r)). The hardness comes from the random, unknown direction and frequency, not from the architecture.
- **§3, decomposition versus end-to-end.** Theorem 3 bounds the signal-to-noise for parity over k "line" images: Var ≤ G²·O(√(k log d / d))^k.
- **§4, ill-conditioning.** Regressing piecewise-linear curves is badly conditioned.
- **§5, flat activations.** Piecewise-flat activations give zero gradients.

## What they find

- Parity, one ReLU layer of width 10d with hinge loss: accuracy falls to chance at d ≈ 30.
- Parity of k line images:
  - End-to-end training fails for k ≥ 3, even at 20k iterations.
  - Adding a per-image auxiliary loss (decomposition) solves it in about 2.5k iterations.
  - The measured gradient signal-to-noise drops by orders of magnitude at k ≥ 3, until float noise can swamp it. The authors stress this is not a local-minimum problem.
- §4, piecewise-linear curves:
  - A fully connected encoder needs Ω(n^3.5) iterations and a convolutional one Ω(n³).
  - Explicit preconditioning of the conv layer gains orders of magnitude.
  - A deeper autoencoder does not help.
- §5: smoothing the step activation with sigmoids is slow, sensitive to bias initialization, and sometimes fails completely. End-to-end regression with a ReLU MLP also struggles. A non-gradient "forward-only" update (GLM-tron style) solves it efficiently, with guarantees for one layer.

## Implementation details

- The fixes that work all change the problem rather than the step count:
  - auxiliary supervision of intermediate quantities (decomposition)
  - architectural or explicit preconditioning
  - replacing gradient through flat parts with a surrogate update
- Measure gradient signal-to-noise (the variance of the gradient across targets relative to its norm) as a diagnostic.

## Caveats

- Synthetic toy problems with worst-case-over-target arguments; the bounds concern families of targets, not a single fixed one.
- Nothing about flow matching, regression to an average, or periodic latent parameters.
- The theory for linear-periodic functions concerns high-frequency periodicity, meaning large ‖v‖.
