# Representation Entanglement for Generation: Training Diffusion Transformers Is Much Easier Than You Think

Wu et al., 2025, NeurIPS 2025 oral · [arXiv:2507.01467](https://arxiv.org/abs/2507.01467)

**Evidence: A.** Peer-reviewed (Neural Information Processing Systems); 112 citations in 1.3 yr
(Semantic Scholar, 2026-10-19).

## What they do

REG. Append a noised DINOv2 class token and denoise it jointly.

## What they find

- SiT-B: vanilla 33.0, REPA 24.4, REG 15.2.
- Choice of extra token: learnable 23.31, latent average 24.12, DINOv2 features 16.86, DINOv2 class token 15.22.
- Without the alignment loss: 26.67.

## Implementation details

Jointly *generate* a meaningful global summary.

## Caveats

The gain depends on the summary carrying meaning.
