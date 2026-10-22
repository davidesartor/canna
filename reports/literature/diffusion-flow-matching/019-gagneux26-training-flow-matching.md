# Training Flow Matching: The Role of Weighting and Parameterization

Gagneux, Martin, Gribonval, Massias, 2026, arXiv · [arXiv:2603.06454](https://arxiv.org/abs/2603.06454)

**Evidence: C.** Not peer-reviewed; 4 citations in 0.6 yr (Semantic Scholar, 2026-10-19). Its
claims are the authors' alone: a hypothesis, not a result.

## What they do

Every FM objective is written as E[w_t ‖D(x_t) − x1‖²] for a denoiser D, separating the loss weight w_t from the network parametrisation.
- Denoiser (x1-loss): w = 1. Velocity loss with D = x + (1−t)N: w = 1/(1−t)². Noise loss: w = t²/(1−t)² (the SNR). "Classical": 1/t².
- In a linear-Gaussian toy, the optimal inverse-variance weight is 1/τ² + t²/(1−t)².

## What they find

- CIFAR-10 FID with velocity parametrisation: w_noise 3.71, w_vel 3.90, w_den 10.29, w_classic 116.18. Other weights: 1/(1−t) gives 6.11, 1/(1−t)³ gives 8.76. CelebA-64: w_noise 3.97, w_vel 4.45.
- Parametrisation: velocity wins for U-Nets and fine-patch ViTs (ViT/4, ViT/8). The denoiser wins for ViT/16 (PSNR gap 79 vs 322) and for MLPs, and its advantage grows as the intrinsic dimension m falls (Fourier-32 data with m = 4/8/16).
- Data size: the denoiser wins at 10k images; velocity wins at 50k and 100k (FID 6.89 vs 8.98, 4.45 vs 9.81).
- Denoising PSNR correlates with FID, so it is usable as a cheap proxy.

## Implementation details

t uniform, batch 128, EMA. Weight and parametrisation are swept independently.

## Caveats

Seeds and error bars not stated; some table labels are ambiguous. Images only. Euclidean Gaussian base.
