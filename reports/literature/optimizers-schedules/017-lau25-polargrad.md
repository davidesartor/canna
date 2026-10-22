# PolarGrad: A Class of Matrix-Gradient Optimizers from a Unifying Preconditioning Perspective (Lau et al., 2025)

[arXiv:2505.21799](https://arxiv.org/abs/2505.21799), preprint.

**Evidence: B.** Not peer-reviewed; 54 citations in 1.4 yr (Semantic Scholar, 2026-10-19).

## What they do

- Frame Muon as a preconditioner built from the polar decomposition of the gradient.
- PolarGrad: Muon's orthogonalised update, scaled by the gradient's nuclear norm.

## What they find

- Muon's step norm does not depend on the gradient's size, so it does not shrink as the
  gradient vanishes. Nuclear-norm scaling restores that.
- PolarGrad beats Adam and Muon in their tests.

## Limits

- Small-scale experiments. No diffusion or flow-matching test.
