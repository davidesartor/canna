# Analysis of deep complex-valued convolutional neural networks for MRI reconstruction and phase-focused applications

Cole, Cheng, Pauly & Vasanawala, 2021, Magnetic Resonance in Medicine 86(2) ·
[arXiv:2004.01738](https://arxiv.org/abs/2004.01738) ·
[DOI:10.1002/mrm.28733](https://doi.org/10.1002/mrm.28733)

**Evidence: A.** Peer-reviewed (MRM); 139 citations in ~5.5 yr (Semantic Scholar,
2026-10-19; Europe PMC lists 70).

## What they do

End-to-end complex-valued CNNs (complex convolution + complex activations) versus
**2-channel (Re, Im) real CNNs at equal trainable parameter count**, for accelerated MRI
reconstruction, where the data are complex k-space/images whose phase carries physics
(fat–water chemical shift, flow velocity). Two architectures (an unrolled
iterative-soft-shrinkage network with data-consistency blocks, and a U-Net), a sweep over
width and depth, five activations, and three datasets (knee, dual-echo body, 2D
phase-contrast flow).

## What they find

Complex beat real at matched parameters on every architecture and dataset (mean ± sd):

| Data / model | PSNR (C vs R) | NRMSE (C vs R) | SSIM (C vs R) |
|---|---|---|---|
| Knee, unrolled | 36.08 vs 34.18 | 0.13 vs 0.16 | 0.92 vs 0.90 |
| Knee, U-Net | 35.28 vs 35.00 | 0.16 vs 0.17 | 0.90 vs 0.90 |
| Body, unrolled | 29.51 vs 28.54 | 0.17 vs 0.22 | 0.78 vs 0.77 |
| Phase-contrast, unrolled | 33.50 vs 32.27 | 0.21 vs 0.22 | 0.88 vs 0.86 |

- Gain ≈ 1–2 dB PSNR for the unrolled net, ≈ 0.3 dB for the U-Net.
- **Phase tasks**: flow error 6.27 % (complex) vs 7.23 % (real); peak velocity 23 % closer to
  ground truth (one patient: 124.6 cm/s complex, 270.8 real, 107.6 truth). Fat–water
  pixel error only 0.32 % better, but the real model shows more fat/water swaps
  qualitatively. Phase maps show fewer wrapping artefacts (qualitative only).
- **Activations** (knee, unrolled, 256 maps): CReLU best (PSNR 32.32, NRMSE 0.31), zReLU
  nearly equal, cardioid and "complex ReLU" close, **modReLU worst** of the complex ones
  (30.67); real ReLU baseline 30.47.
- Authors' explanation: a 2-channel real conv applies different weights to Re and Im and
  "potentially alter[s] the phase", whereas complex conv "preserves the phase of the input
  data"; and at equal parameters the complex net gets more feature maps.

## Implementation details

- Parameter matching: a complex conv layer has half the parameters of a real one with the
  same number of channels, so the complex net gets ≈ √2 more maps (complex 8, 16, 32, 64,
  128, 256 vs real 6, 12, 22, 46, 90, 182; 128 vs 90 in the main runs).
- L1 loss, Adam (β = 0.9, 0.999, lr 1e-3), batch 2–3, TensorFlow 1.14. Initialisation,
  precision and epoch count not reported.

## Caveats

- Image-to-image regression with very high per-pixel SNR in the fully sampled targets; the
  inputs are undersampled, not noise-dominated.
- Gains are modest and architecture-dependent (U-Net ≈ tie on SSIM); fat–water numeric gain is
  negligible and the phase claims are largely qualitative.
- Matching gives the complex net more channels; the paper does not separate "more maps" from
  "complex arithmetic".
- Note CReLU (not phase-equivariant) won over modReLU (equivariant): phase-preserving
  linear layers mattered more than phase-equivariant activations here.
