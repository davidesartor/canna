# Adaptive Non-uniform Timestep Sampling for Accelerating Diffusion Model Training

Kim, Ki, Shim, Lee, 2024; arXiv v2 Oct 2025 · [arXiv:2411.09998](https://arxiv.org/abs/2411.09998)

**Evidence: B.** Peer-reviewed (Computer Vision and Pattern Recognition); 10 citations in 1.9
yr (Semantic Scholar, 2026-10-19).

## What they do

Per-t stochastic gradient variance differs up to ~100× across t (high at the clean end for linear/cosine/quadratic schedules under L_DDPM), and with an Adam-like optimiser the iterations to converge scale with that variance (Thm 1: K ≥ O(Var/ε²)). But subproblems are strongly coupled: training only t∈[0,200) raises the loss on untrained t by more than it lowers it on trained t (Fig 4), so "sample ∝ gradient variance" fails. Instead learn a sampler π_φ(a,b|x₀), t ~ Beta(a,b), by REINFORCE: φ ← φ + γ·Δ̃_k^t·∇_φ log π_φ, where the reward Δ_k^t = (1/T)Σ_τ [L_τ(θ_k) − L_τ(θ_{k+1})] is the drop in the full VLB after the step on t. Δ is approximated on |S|=3 timesteps chosen by F-statistic feature selection over a queue (|Q|=20) of full per-τ δ vectors computed on one x₀; φ updated every f_S=40 steps; entropy bonus 1e−2.

## What they find

CIFAR-10 DDPM FID (best so far at 0.2M/0.4M/0.6M/0.8M/1.0M), linear schedule: DDPM 8.16/6.03/5.37/4.28/3.19; P2 6.45/…/4.70; Min-SNR 4.40/3.51/3.28/3.16/3.16; log-normal (EDM) 5.38/…/3.60; SpeeD 4.08/…/3.57; **ours 3.99/3.21/3.10/2.94/2.94**. Cosine (0.2/0.4/0.6M): DDPM 3.74/3.14/3.14; Min-SNR 3.51/3.07/3.07; ours 3.87/3.06/3.00. Quadratic: Min-SNR degrades to 4.17 (worse than uniform 3.01); ours 2.95. CelebA-HQ 256 LDM 100/300 epochs: LDM 12.87/8.72; P2 10.69/6.88; Min-SNR 10.87/6.83; ours 10.94/6.62. ImageNet-256 U-ViT: Min-SNR 11.5→4.54 vs Min-SNR+ours 11.3→4.33 at 0.2M→1.0M. Ablation (Table 4): sampling ∝ gradient variance gives 21.3→11.3 vs ours 3.99→2.94 (much worse than uniform). Overhead ≈1.41× per iteration in theory, ~1.5× wall clock in practice; Fig 1 claims better FID within the baseline's convergence time and clearly better at 1.2×.

## Implementation details

Beta(a,b) parametrisation enforces smoothness across neighbouring t; heuristic weights (Min-SNR) can be folded in and the learned sampler compensates; learned distributions over training end up concentrating at low t (linear) or both ends (cosine), roughly following gradient variance despite the variance-proportional sampler failing.

## Caveats

gains are small at convergence on cosine (3.00 vs 3.07 Min-SNR) and early-iteration gains are often negative vs heuristics; reported "lowest FID so far" per budget; ~1.5× compute per step not always netted out; only discrete-time ε-prediction DDPM/LDM, not FM or score-based EDM (stated limitation).
