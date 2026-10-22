# Harmonic Self-Conditioned Flow Matching for Joint Multi-Ligand Docking and Binding Site Design

Stärk, Jing, Barzilay, Jaakkola, 2023/v4 2024; ICML 2024 per page keywords · [arXiv:2310.05764](https://arxiv.org/abs/2310.05764)

**Evidence: B.** Peer-reviewed (International Conference on Machine Learning); 48 citations in
3.0 yr (Semantic Scholar, 2026-10-19).

## What they do

Flow matching on ligand atom coordinates from a "harmonic" prior (Gaussian shaped by the bond-graph Laplacian). The network predicts x1, not velocity. Self-conditioning: v(x_t, x̃1, t), where x̃1 is **a prior sample on 50% of training steps** (not zero), and otherwise sg of the model's own first-pass x1 prediction. At inference the first input is a prior sample, then the previous prediction. This is described as AlphaFold2-style recycling. FlowSite extends it to residue types.

## What they find

PDBBind radius-pocket, sequence split, % RMSD<2Å (best-of-5): full 20.5 (34.5). **No self-conditioning 14.3 (29.8).** Gaussian prior 17.0. Velocity prediction **11.9** (28.8). No refinement loss 9.8. Standard TFN 13.7. σ=0 18.3. Against diffusion (product-space): time-split distance-pocket 42.8 vs 20.8. Multi-ligand MOAD 44.4 vs 39.7 (EigenFold). Self-conditioning costs about 12.8% training time.

## Implementation details

Use a prior draw rather than zeros as the "no self-conditioning" input. Predict x1 rather than v. Add an auxiliary refinement loss on intermediate positions.

## Caveats

Docking metrics (RMSD success) are not distributional or calibration metrics. One domain only.
