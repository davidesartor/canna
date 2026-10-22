# Equivariance with Learned Canonicalization Functions

Kaba et al., 2022, ICML 2023 · [arXiv:2211.06489](https://arxiv.org/abs/2211.06489)

**Evidence: A.** Peer-reviewed (International Conference on Machine Learning); 140 citations in
3.9 yr (Semantic Scholar, 2026-10-19).

## What they do

Map the input to a canonical frame with h(y), run an unconstrained network, and map the output back. This gives exact equivariance with any backbone.

## What they find

Learned canonicalizers match equivariant networks on its benchmarks.

## Implementation details

The canonicalizer must be continuous.

## Caveats

Continuity matters when the frame is ambiguous.
