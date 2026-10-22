# Equivariant Flow Matching with Hybrid Probability Transport for 3D Molecule Generation (EquiFM)

Song et al., 2023; NeurIPS 2023 · [arXiv:2312.07168](https://arxiv.org/abs/2312.07168)

**Evidence: B.** Peer-reviewed (Neural Information Processing Systems); 73 citations in 2.9 yr
(Semantic Scholar, 2026-10-19).

## What they do

coordinates on equivariant-OT straight paths (Hungarian + Kabsch alternation), categorical atom features on a VP path chosen so their information decay matches I(x_t,h0); paths per modality can differ since conditionals factorise.

## What they find

QM9 molecule stability 88.3 (EDM 82.0). Ablation: EOT→plain OT 88.3→84.9; VP coordinate path 81.6; categorical VP_cos 84.7, VP_poly 83.4, OT path for categorical 77.1. ~210 NFE (dopri5) vs EDM 1000.

## Caveats

single seed std small; molecules only.
