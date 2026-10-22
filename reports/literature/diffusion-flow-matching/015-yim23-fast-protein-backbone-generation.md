# Fast protein backbone generation with SE(3) flow matching

FrameFlow; Yim et al., 2023, arXiv/workshop · [arXiv:2310.05297](https://arxiv.org/abs/2310.05297)

**Evidence: A.** Not peer-reviewed; 163 citations in 3.0 yr (Semantic Scholar, 2026-10-19).

## What they do

SE(3) FM, uniform SO(3) × centred Gaussian R³. The network predicts x̂1 and r̂1. The vector fields are log_{x_t}(x̂1)/(1−t) and log_{r_t}(r̂1)/(1−t). The loss is on the vector fields with 1/(1−t)² weighting.

## What they find

- Designability (scRMSD<2Å): **0.77 at 100 ODE steps vs FrameDiff 0.42 (500 SDE steps)**, 0.81 at 500 steps, 0.33 at 10 steps.
- GENIE 0.22.
- FrameDiff ODE 0.26 at 100 steps.
- **All ablations (linear rotation schedule "performed poorly", Kabsch pre-alignment, IGSO3 prior) are qualitative, with no numbers.**

## Implementation details

- **Rotation inference schedule κ(t) = e^{−ct}, c=10** (5 also fine), i.e. the rotation velocity becomes c·log_{r_t}(r̂1).
- **1/(1−t)² clipped to 1/(1−min(t,0.9))²** because the loss exploded near t≈1.
- t~U[0,1−1e-3].
- Rotation loss weight 0.5.
- **Training prior IGSO3(σ=1.5) instead of uniform** (sampling still uniform), reported to improve results.
- Kabsch pre-alignment of noise to data; self-conditioning; Adam 1e-4.

## Caveats

No numeric ablations. Small dataset (3,938 SCOPe proteins). A workshop-style paper.
