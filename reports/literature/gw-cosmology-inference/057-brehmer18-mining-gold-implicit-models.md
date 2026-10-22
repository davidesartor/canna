# Mining gold from implicit models to improve likelihood-free inference

Brehmer, Louppe, Pavez, Cranmer, 2018; PNAS 117, 5242 (2020) · [arXiv:1805.12244](https://arxiv.org/abs/1805.12244)

**Evidence: A.** Peer-reviewed (Proceedings of the National Academy of Sciences of the United
States of America); 218 citations in 8.5 yr (Semantic Scholar, 2026-10-19).

## What they do

From simulator latents z, compute the joint score t(x,z|θ) = ∇θ log p(x,z|θ) and the joint ratio r(x,z|θ0,θ1).
- Key lemma: the MSE regression of ĝ(x) on g(x,z) is minimized by E_{p(z|x)}[g], so regressing on joint quantities yields the intractable marginal ratio and score.
- Losses built on this:
  - ROLR: ratio regression.
  - RASCAL / CASCAL: ratio or classifier loss + α·score MSE.
  - SCANDAL: flow MLE + α·score MSE.
  - SALLY / SALLINO: a learned score at θ_ref used as locally sufficient statistics (local exponential-family expansion), followed by low-dimensional density estimation.

## What they find

- Galton board: SCANDAL and RASCAL are dramatically more sample-efficient than NDE or the likelihood-ratio trick. ROLR reaches the same asymptote more slowly.
- Lotka-Volterra (narrow θ range ±0.01 in log space): SCANDAL is better on log r, but the score term's variance **hurts log p density estimation**, more so at larger α (α = 0.01 vs 0.1). There is a trade-off between density and ratio quality.
- Higgs → 4ℓ (x ∈ ℝ⁴², 2 parameters): RASCAL needs about **100× less training data** than CARL. SALLY and SALLINO need even less but plateau, because the local model breaks down away from θ_ref.
- Augmented data helps for nearby θ and is weak for distant θ, where the joint ratio has huge variance.

## Implementation details

α ≈ 0.01. Gradients are most valuable for distinguishing small parameter differences, i.e. narrow posteriors.

## Caveats

Ratio and likelihood estimation, not NPE. Particle-physics details are in a companion paper.
