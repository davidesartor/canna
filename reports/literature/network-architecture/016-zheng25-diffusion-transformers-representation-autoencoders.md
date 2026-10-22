# Diffusion Transformers with Representation Autoencoders

Zheng et al., 2025 · [arXiv:2510.11690](https://arxiv.org/abs/2510.11690)

**Evidence: A.** Not peer-reviewed; 314 citations in 1.0 yr (Semantic Scholar, 2026-10-19).

## What they do

Diffusion in a high-dimensional representation latent (DINOv2), with a wide, shallow "DiT^DH" head on (x_t, z_t, t).

## What they find

- Head shapes: 2 layers × 2048 (26.78 GFLOPs) gives FID 2.16; 6 × 1152 gives 2.36; 4 × 2048 gives 2.31.
- The backbone must be at least as wide as the token dimension. Overfitting one image with DINOv2-B (n = 768): DiT-S reaches loss 0.52, DiT-B 0.024. Doubling depth does not help.
- A dimension-dependent time shift α = √(m/n) takes FID 23.08 → 4.81.

## Implementation details

Keep width ≥ token dimension, and shift the time schedule with dimension.

## Caveats

**Negative:** on SD-VAE latents DiT^DH-XL is *worse*, 7.13 → 11.70. It helps only for high-dimensional tokens.
