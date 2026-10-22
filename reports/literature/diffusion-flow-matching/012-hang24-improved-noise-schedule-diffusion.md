# Improved Noise Schedule for Diffusion Training

Hang, Gu, Geng, Guo, 2024, arXiv v2 Nov 2024 · [arXiv:2407.03297](https://arxiv.org/abs/2407.03297)

**Evidence: B.** Peer-reviewed (IEEE International Conference on Computer Vision); 59 citations
in 2.3 yr (Semantic Scholar, 2026-10-19).

## What they do

Choosing a schedule is the same as choosing a density p(λ) over log-SNR: t = 1 − CDF(λ), λ(t) = CDF⁻¹(1−t). Table 1 formulas: cosine p(λ)=sech(λ/2)/(2π), λ=2 log cot(πt/2); **Laplace** p(λ)=e^{−|λ−μ|/b}/(2b), λ(t)=μ − b·sgn(0.5−t)·ln(1−2|t−0.5|); Cauchy λ=μ+γ tan(π(1−2t)/2); cosine-shifted λ=μ+2 log cot(πt/2) (= Hoogeboom shift); cosine-scaled λ=(2/s) log cot(πt/2). VP: α²=σ(λ), σ²=σ(−λ). Appendix A.4: SD3 logit-normal(m,s) on the FM time is exactly **λ ~ N(−2m, 4s²)**, a Gaussian in log-SNR centred at λ=−2m. The paper argues that changing p(λ) (where compute goes) beats changing w(λ) (how much loss counts), even though they are equivalent in expectation. Inference is aligned: all models sample at the cosine-schedule λ grid (DDIM 50 steps), so only training differs.

## What they find

DiT-B, ImageNet-256 latent, 500k iterations, batch 256, FID-10k at CFG 1.5/2.0/3.0 (Table 3): cosine 17.79/10.85/11.06; EDM 26.11/15.09/11.56; FM-OT (uniform-t rectified flow) 24.49/14.66/11.98; Min-SNR 16.06/9.70/10.43; Soft-Min-SNR 14.89/9.07/10.66; cosine-shifted (μ=1) 19.34/11.67/11.13; cosine-scaled (s=2) 12.74/8.04/11.02; Cauchy(0,0.5) 12.91/8.14/11.02; Laplace(0,0.5) 16.69/9.04/**7.96** (−2.89 vs best cosine 10.85, −26.6%). Prediction targets at 500k (Table 4): x₀ 11.16→8.48, v 11.06→7.96, ε 10.46→9.53. ImageNet-512: cosine 11.91 → Laplace b=0.75 9.09. Laplace b sweep {0.25,0.5,1,2,3}: b=0.5 best. Cauchy: centred μ=0 beats μ=±1 (CFG 2.0: 8.93 vs 10.38/10.19). Cosine-shifted μ=±1 is worse than unshifted at 256². Cosine-scaled 1/s sweep {1.3,1.1,0.5,0.25}: CFG2 23.38/12.98/8.04/8.64, so concentrating too much hurts. Same density as weight vs as sampling (Table 9): cosine-ply n=2 sampling 7.98, as loss weight 8.88, Min-SNR 9.70, Soft-Min-SNR 9.07. CelebA-64 and CC12M also improve (CC12M FID 58.36→54.35).

## Implementation details

DDIM starts at t_max=0.99 because the Laplace schedule under-trains extreme noise. Polynomial t density p(t)∝min(t,1−t)^n (cosine-ply) with n=2 ≈ Laplace b=1. Fine-tuning per t-quarter: the two middle bins matter most for FID.

## Caveats

Single small backbone (DiT-B), 500k iterations, FID-10k and CFG-selected; the "best" schedule changes with CFG scale (Laplace is not best at CFG 1.5); the λ=0 optimum is specific to data normalisation (unit-variance latents) and to VP schedules; no comparison against logit-normal FM itself.
