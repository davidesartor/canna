# Reconstruction vs. Generation: Taming Optimization Dilemma in Latent Diffusion Models

Yao et al., 2025, CVPR 2025 · [arXiv:2501.01423](https://arxiv.org/abs/2501.01423)

**Evidence: A.** Peer-reviewed (Computer Vision and Pattern Recognition); 447 citations in 1.8
yr (Semantic Scholar, 2026-10-19).

## What they do

LightningDiT / VA-VAE. A cumulative recipe of modern DiT hygiene.

## What they find

Starting from 19.50: rectified flow 17.20, logit-normal 13.99, cosine-velocity loss 12.52, SwiGLU 10.10, RMSNorm 9.25, RoPE 7.13.

## Implementation details

Each step is ablated cumulatively.

## Caveats

Gains are measured in sequence, not independently.
