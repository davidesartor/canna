# Equivariance with Learned Canonicalization Functions

Kaba, Mondal, Zhang, Bengio, Ravanbakhsh, 2022; ICML 2023 · [arXiv:2211.06489](https://arxiv.org/abs/2211.06489)

**Evidence: A.** Peer-reviewed (International Conference on Machine Learning); 140 citations in
3.9 yr (Semantic Scholar, 2026-10-19).

## What they do

φ(x) = h'(x) f(h(x)⁻¹x): a small equivariant canonicaliser h outputs a group element, an unconstrained f does the work; universality if h continuous; h direct (equivariant net) or by argmin of a learned energy.

## What they find

Rot-MNIST error CN(p64)-CNN 1.99% vs G-CNN 2.28%, aug. CNN 3.30%; N-body MSE 0.0043 vs EGNN 0.0071, frame averaging 0.0057, frozen canonicaliser 0.0085; ModelNet40 SO(3)/SO(3) 90.0 vs VN-DGCNN 90.2. Optimisation-based canonicaliser worse (stuck in flat regions).

## Caveats

authors: "no guarantees the canonicalization function is smooth" — discontinuities can hurt generalisation; no permutation experiments.
