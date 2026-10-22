# PixelDiT: Pixel Diffusion Transformers for Image Generation

Yu et al., 2025, CVPR 2026 · [arXiv:2511.20645](https://arxiv.org/abs/2511.20645)

**Evidence: A.** Peer-reviewed; 91 citations in 0.9 yr (Semantic Scholar, 2026-10-19).

## What they do

A 26-block encoder plus a narrow (width-16) pixel decoder modulated **per pixel** by encoder features.

## What they find

Per-pixel adaLN 3.50 → 2.36. RMSNorm + RoPE 9.84 → 8.53.

## Implementation details

Per-token adaLN parameters from features, instead of one global c.

## Caveats

Single paper.
