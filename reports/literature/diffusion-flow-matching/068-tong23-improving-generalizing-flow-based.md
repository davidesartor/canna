# Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport

Tong et al., 2023; TMLR 2024 · [arXiv:2302.00482](https://arxiv.org/abs/2302.00482)

**Evidence: A.** Peer-reviewed (Trans. Mach. Learn. Res.); 1085 citations in 3.7 yr (Semantic
Scholar, 2026-10-19).

## What they do

CFM regresses v_θ(t,x) onto conditional target u_t(x|z)=x1−x0 on x_t ~ N(t x1+(1−t)x0, σ²). I-CFM: z=(x0,x1) independent. OT-CFM: (x0,x1) drawn from the exact W2 plan computed per minibatch (POT exact EMD). SB-CFM: entropic plan with λ=2σ² and Brownian-bridge conditionals N(·, t(1−t)σ²); OT-CFM is the ε→0 limit, I-CFM ε→∞.

## What they find

2D normalized path energy (NPE) OT-CFM 0.018–0.087 vs I-CFM 0.22–2.74 (e.g. moons→8g 0.053 vs 2.738); W2 roughly equal (1.262 vs 1.284 on N→8g). CIFAR-10 (adaptive dopri5): OT-CFM FID 3.58 @134 NFE vs I-CFM 3.66 @146 NFE; at 100 Euler NFE 4.44 vs 4.46 — essentially tied; the gain is only visible at small NFE (figure only). Single-cell EMD OT-CFM 0.88/0.79/0.94 vs I-CFM 0.97/0.87/1.09. Energy-based 10-D funnel: with Euler N=10, log Ẑ bias OT-CFM −0.039 vs CFM 0.281. Objective variance ≥10× lower for OT/SB-CFM (fig only). Batch-size ablation (App. D): NPE drops fast and plateaus at batch ≈64 on 2D; moons→8g needs larger.

## Implementation details

exact EMD per batch (<1% overhead on CIFAR); σ≤1 fine, σ>1 underfits; SB-CFM unstable at small σ; CIFAR "improved" recipe: 128-ch UNet, dropout 0.1, constant lr 2e-4, grad-clip 1, EMA 0.9999, batch 128, 400k steps, σ=0 (most of the FID gain vs the original FM paper comes from this recipe, not from OT).

## Caveats

authors state minibatch OT "can incur error in high dimensions"; image-scale gains vs I-CFM are nil at moderate NFE; no conditional experiment; OT-CFM slower to train in wall-clock on 2D (1.48 vs 1.05 ×10³ s).
