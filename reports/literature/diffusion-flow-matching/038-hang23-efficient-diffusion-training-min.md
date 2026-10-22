# Efficient Diffusion Training via Min-SNR Weighting Strategy

Hang et al., 2023, ICCV 2023 · [arXiv:2303.09556](https://arxiv.org/abs/2303.09556)

**Evidence: A.** Peer-reviewed (IEEE International Conference on Computer Vision); 345
citations in 3.6 yr (Semantic Scholar, 2026-10-19).

## What they do

Treat each noise level as a task in multi-task learning. Fine-tuning on a t-bin helps neighbouring bins but hurts distant ones (Fig 2: gradient conflict). The Pareto-optimal weights (min‖Σw_t∇L_t‖² + λΣw_t², solved by Frank-Wolfe or softmax-parametrised gradient descent) are noisy and expensive. The fixed proxy: in x₀-mse, w_t = min{SNR(t), γ}, with SNR=α²/σ². Equivalent forms: in ε-mse w = min{γ/SNR, 1}; in v-mse w = min{SNR,γ}/(SNR+1). Identities: ε-mse = SNR·x₀-mse; v-mse = (SNR+1)·x₀-mse.

## What they find

γ ablation (FID, ImageNet-256): ViT x₀ 4.98/4.92/5.34/5.45 for γ=1/5/10/20; ViT ε 4.89/4.84/4.94/5.41; UNet x₀ 4.49/4.28/4.32/4.37; UNet ε 4.30/4.14/4.14/4.12, so γ=5 is the default and the method is robust for γ≤20. UNet table: x₀ baseline 25.93→8.33 over 200k→1M iterations, versus Min-SNR-5 at 7.99→4.28; ε baseline 8.55→4.21 versus 7.32→4.14 (so most of the gain is for x₀-prediction; for ε-prediction the gain is small). Constant-weight ε and Max-SNR ε diverge. Pure SNR weighting (focus on clean end) is worst. Unweighted per-bin loss: constant weighting is good at high noise and bad at low noise, SNR weighting the opposite; Min-SNR is lowest in all bins. SOTA: ImageNet-256 FID 2.06 (ViT-XL, cfg 1.5, ~7M iterations, x₀); 2.08 at 2.1M iterations with ε (3.3× faster than DiT); CelebA-64 1.60 (UNet); ImageNet-64 2.28 (ViT-L).

## Implementation details

cosine schedule, T=1000, AdamW lr 1e−4, ViT betas (0.99, 0.99), EMA 0.9999, EDM Heun sampler. On EDM: multiply EDMLoss by min{SNR,5}/SNR.

## Caveats

weighting only, the time density stays uniform in discrete t (cosine schedule); the "Pareto" justification is heuristic; all evidence is image FID.
