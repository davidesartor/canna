# Simpler Diffusion (SiD2): 1.5 FID on ImageNet512 with pixel-space diffusion

Hoogeboom et al., 2024 · [arXiv:2410.19324](https://arxiv.org/abs/2410.19324)

**Evidence: B.** Not peer-reviewed; 83 citations in 2.0 yr (Semantic Scholar, 2026-10-19).

## What they do

A simplified pixel-space U-ViT recipe.

## What they find

- Residual U-ViT skips are about 0.2 FID worse than blockwise skips when small, and on par when large.
- Spending FLOPs beats adding parameters: 2×2 patching gives 1.54 vs 1.9 for 2× parameters.
- Asymmetric E3-D7 beats E3-D3 and E5-D1 (qualitative).
- FID 1.48 at 512, 1.38 at 256.

## Implementation details

Prefer per-token FLOPs over parameters.

## Caveats

Images only.
