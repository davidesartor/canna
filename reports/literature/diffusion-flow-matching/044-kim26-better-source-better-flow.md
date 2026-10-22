# Better Source, Better Flow: Learning Condition-Dependent Source Distribution for Flow Matching

Kim, Park, Jeon, Kim, 2026; arXiv · [arXiv:2602.05951](https://arxiv.org/abs/2602.05951)

**Evidence: C.** Not peer-reviewed; 9 citations in 0.7 yr (Semantic Scholar, 2026-10-19). Its
claims are the authors' alone: a hypothesis, not a result.

## What they do

source N(μ_φ(c), diag σ²_φ(c)) from a 77M Perceiver on text embeddings, trained jointly with the flow under the FM loss + λ_VarReg·KL(N(μ,σ²)‖N(μ,I)) (variance pinned to 1, mean free) + λ_align·E[1−cos(x0,x1)]; λ_VarReg=5, λ_align=1. Rationale: reduce intrinsic Var(Δ|x_t).

## What they find

ImageNet-256 T2I (RAE-DINOv2 latents, 100k it): Gaussian 3.036 → **no reg: NaN (collapse)** → KL reg 2.904 → VarReg 2.765 → VarReg+MSE align 2.942 → VarReg+cos align 2.453. Across backbones: LightningDiT 3.72→3.04, MMDiT 3.41→3.05. Up to 3.01× faster FID convergence. 50→3 steps FID degradation 8.75 vs 12.47. Beats C2OT, CPD, CAR-Flow, CrossFlow (2.93–3.37). At 1.3B GenEval 0.77→0.80. Gains much smaller in SD-VAE latents.

## Implementation details

stop-grad on target velocity can make source variance explode; when a condition maps to several separated modes the source mean averages and gains vanish.

## Caveats

images only; FID-level gains; target-space dependent.
