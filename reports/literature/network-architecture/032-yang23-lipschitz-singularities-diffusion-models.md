# Lipschitz Singularities in Diffusion Models

Yang et al., 2023, ICLR 2024 · [arXiv:2306.11251](https://arxiv.org/abs/2306.11251)

**Evidence: B.** Peer-reviewed (International Conference on Learning Representations); 30
citations in 3.3 yr (Semantic Scholar, 2026-10-19).

## What they do

The network's Lipschitz constant in t blows up near the data end. E-TSDM shares the conditioning over sub-intervals there.

## What they find

FFHQ FID 9.50 → 6.62.

## Implementation details

Quantize or share t near the singular end.

## Caveats

ε-prediction setting.
