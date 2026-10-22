# Consistency Models Made Easy

Geng et al., 2024, ICLR 2025 · [arXiv:2406.14548](https://arxiv.org/abs/2406.14548)

**Evidence: A.** Peer-reviewed (International Conference on Learning Representations); 163
citations in 2.3 yr (Semantic Scholar, 2026-10-19).

## What they do

Write the consistency condition as df/dt=0 + boundary; discretise as d(f_θ(x_t), f_{sg θ}(x_r)) with shared noise ε, x_t=x0+tε, r<t. With r=0 this *is* diffusion (denoising) training, so a pretrained diffusion model is a CM with Δt=t; Easy Consistency Tuning (ECT) starts from it and shrinks Δt→dt with a continuous mapping r/t = 1 − n(t)/q^{⌊iters/d⌋}, n(t)=1+k·sigmoid(−bt) (k=8,b=1), r clamped ≥0. "Curse of consistency": 1-step error ≤ N·e_max, so tiny Δt from scratch accumulates errors and trains slowly/unstably. Loss weight split into timestep weight w̄(t) and adaptive per-sample weight w(Δ)=1/(‖Δ‖²+c²)^{1/2} (the gradient of pseudo-Huber), keeping plain ℓ2.

## What they find

CIFAR-10 2-step FID 2.20 after 100k iters (1 h on one A100 claimed for 2.73 version), 2.11 at 400k; 1-step 3.60 (iCT-deep 2.51 still better at 1 step). ImageNet-64 ECM-XL 2-step 1.67, 1-step 2.49 with 102.4M images. Tuning costs 0.6–1.9% of pretraining. Ablation (Tab. 6, ImageNet-64, EDM2-S init): iCT recipe 21.09/4.39 (1/2-step) → + continuous-time schedule 14.34/4.33 → + dropout 0.4 9.28/3.22 → + w̄=1/t²+1/σ_d² 5.51/3.18. Adaptive weighting (Tab. 3): w(Δ)=1 6.51/3.28 vs ℓ1-style 6.29/3.25 vs ℓ2-style 5.51/3.18. Mapping fn: constant n(t)≡1 gives 4.06 vs 3.86 (CIFAR) and *instability on ImageNet*. Using ECT findings (adaptive w(Δ), p=1/2, w̄=1/t) in *diffusion pretraining* gives >2× faster convergence (FD_DINOv2 150.4 vs 168.2). Scaling: FD_DINOv2 ∝ C^−0.060 (1-step), C^−0.028 (2-step); 2-step smaller model beats 1-step larger one.

## Implementation details

Dropout surprisingly high (0.4–0.5 on ImageNet; 2-step FID 4.53 → 3.24 going 0.1→0.4); share dropout seed across the pair. Best 1-step and 2-step recipes differ (weighting, dropout, EMA). Toy Swiss-roll FM experiment: adaptive weighting w(Δ)=1/(‖Δ‖²+ε)^p in *ordinary flow matching* makes the learned flow much easier to sample in few steps, even p=1. For FM parametrisations they advise starting with the constant mapping function.

## Caveats

Images; the FM-adaptive-weighting claim is a 2-D toy figure only; 1-step quality still behind iCT-deep at equal model size.
