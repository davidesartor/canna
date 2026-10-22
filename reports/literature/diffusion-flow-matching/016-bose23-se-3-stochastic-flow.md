# SE(3)-Stochastic Flow Matching for Protein Backbone Generation

FoldFlow; Bose et al., 2023, ICLR 2024 · [arXiv:2310.02391](https://arxiv.org/abs/2310.02391)

**Evidence: A.** Peer-reviewed (International Conference on Learning Representations); 207
citations in 3.0 yr (Semantic Scholar, 2026-10-19).

## What they do

SO(3)×R³ FM in three variants:
- Base: geodesic interpolant with the closed-form log-map target.
- OT: Riemannian minibatch OT coupling on SO(3).
- SFM: a Brownian-bridge approximation with IGSO3 noise of variance γ²t(1−t) around the geodesic.
The network predicts x̂0 (data at t=0 in their convention) via IPA. Auxiliary losses apply at t<0.25.

## What they find

- Designability: FoldFlow-OT 0.820, SFM 0.716, Base 0.657, FrameDiff-retrained 0.612, RFdiffusion 0.969.
- **Ablation: everything off 0.228 → inference annealing alone 0.648**, + aux 0.657, + OT 0.820, + stochastic 0.716 (more novelty, less designable).
- Toy SO(3) W1 (×1e-2): Base 5.39 (mode shrinkage), OT 4.96, SFM 4.92.

## Implementation details

- **Inference annealing: multiply the rotation velocity by i(t)=c·t, c≈10.** This removes an unwanted rise in rotation-flow norm at the end of sampling and is "critical".
- Rotation loss 0.5 (split into axis and angle terms), translation 1.0, λ_aux=0.25.
- Adam 1e-4, ~600k steps, batch ∝ 1/N².
- The log near angle π is not discussed.
- The appendix notes that FrameDiff-ICML had a stop-gradient bug in its rotation score.

## Caveats

The time convention is inconsistent across algorithms. The annealing c sweep is a figure only. The effect on distributional calibration is not measured.
