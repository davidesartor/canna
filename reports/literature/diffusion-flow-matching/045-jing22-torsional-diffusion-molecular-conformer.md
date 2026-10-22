# Torsional Diffusion for Molecular Conformer Generation

Jing, Corso et al., 2022, NeurIPS 2022 · [arXiv:2206.01729](https://arxiv.org/abs/2206.01729)

**Evidence: A.** Peer-reviewed (Neural Information Processing Systems); 416 citations in 4.4 yr
(Semantic Scholar, 2026-10-19).

## What they do

Score diffusion on the hypertorus of torsion angles. The perturbation kernel is a wrapped normal (lattice sum), with the score precomputed numerically. σ(t) = σ_min^{1−t}σ_max^t with σ_min=0.01π and σ_max=π, and a uniform prior. Loss weight λ(t) = 1/E‖∇log p_{t|0}‖², also precomputed. An SE(3)-invariant, **parity-equivariant** e3nn network outputs one pseudoscalar score per rotatable bond. The reverse step is a wrapped random walk.

## What they find

- GEOM-DRUGS recall coverage 72.7% (mean) / AMR 0.582 Å, vs GeoDiff 42.1% / 0.835.
- Step ablation, AMR-R: 3→0.820, 5→0.698, 10→0.600, 20→0.582, 50→0.578.
- Other ablations (AMR-R mean, baseline 0.582):
  - no parity equivariance 0.928 (≈ random torsions 0.922),
  - train on ground-truth local structure (distribution shift) 0.920,
  - PF-ODE 0.577,
  - first-order irreps 0.605.
- Boltzmann generator ESS (out of 32) at 300K: 4.68 vs AIS-100 2.06.

## Implementation details

Score normalisation by the expected score norm per σ. Exact divergence is feasible in low dimension. Annealed-temperature training for the energy-based version.

## Caveats

The local-structure model is not learned. The score for the periodic coordinate is easy here because of strong symmetry inductive bias.
