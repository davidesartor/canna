# Metric Flow Matching for Smooth Interpolations on the Data Manifold

Kapuśniak et al., 2024, NeurIPS 2024 · [arXiv:2405.14780](https://arxiv.org/abs/2405.14780)

**Evidence: A.** Peer-reviewed (Neural Information Processing Systems); 92 citations in 2.4 yr
(Semantic Scholar, 2026-10-19).

## What they do

Learn the interpolant x_t = (1−t)x0 + t x1 + t(1−t)φ_η(t,x0,x1) by minimising the metric kinetic energy E[ẋᵀG(x)ẋ] (stage 1), then do CFM on those paths (stage 2). It combines with minibatch OT.

## What they find

- Arch EMD: OT-CFM 0.608 → OT-MFM 0.081.
- Sphere-embedded data, midpoint distance off the sphere: 0.519 → 0.085, and EMD 0.525 → 0.340.
- Single-cell W1 (5-dim): Cite 0.882 → 0.724, EB 0.790 → 0.713. At 100-dim, Cite 45.39 → 41.78.
- AFHQ dog→cat latent: FID 41.42 → 37.87.
- OT vs independent coupling matters, e.g. Cite 0.724 vs 0.916.

## Implementation details

LAND σ=0.125, ε=1e-3. RBF κ=0.5–1.5, with (h̃)^8 sharpening. MLPs 3×64 SeLU. 100 Euler steps.

## Caveats

Trajectory-inference / unpaired-translation tasks, not posterior sampling. No ε/κ ablations. LAND may bias paths toward dense regions.
