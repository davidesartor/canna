# SimBIG: Field-level Simulation-Based Inference of Galaxy Clustering

Lemos, Parker et al., 2023, arXiv/PRD · [arXiv:2310.15256](https://arxiv.org/abs/2310.15256)

**Evidence: B.** Peer-reviewed; 8 citations in 3.0 yr (Semantic Scholar, 2026-10-19).

## What they do

- Two-stage pipeline. A 3D CNN regresses the 5 ΛCDM parameters from a 64×128×128 CIC galaxy mesh (k < 0.28) with MSE on (0,1)-normalised θ.
- SWAG weight marginalisation: 10 weight samples × 5 outputs give a 50-D summary.
- An ensemble of 5 NSFs (`sbi`) does NPE on that summary.
- Robustness to forward-model misspecification is enforced by validating on test sets built with a different halo finder and HOD, and a different N-body code.

## What they find

- Ω_m = 0.267 (−0.029/+0.033) and σ8 = 0.762 ± 0.036, 1.76× and 1.92× tighter than SimBIG P_ℓ. σ8 is 2.65× tighter than PT P_ℓ.
- H0: the abstract says 64.5 ± 3.8 but Results says 63.1 ± 4.1, an internal inconsistency.
- TARP ECP on 5,180 held-out sims is "perfectly" diagonal.
- Mock challenge: the (μ − θ_fid)/σ distributions agree across TEST0, TEST1 and TEST2.
- Ablation (footnote 5): the one-step approach, with the CNN as an embedding net trained jointly with the NPE, gave "significantly weaker" constraints than the two-stage MSE-then-NPE approach.

## Implementation details

- Data: 2,518 LHC N-body sims × 10 HOD draws = 25,180 (not i.i.d.; "we expect regularization to deal with" it). Split 20,000 train / 5,180 validation.
- CNN: 5 blocks of [3³ conv, ReLU, maxpool, BatchNorm] + 3 FC layers.
  - Dropout 0.15 on conv and 0.4 on FC; ℓ2 λ = 0.0275.
  - SGD with momentum 0.9, batch 32, OneCycleLR with max lr 0.01; random flips; early stopping at patience 20; ~8 h on one A100.
- Optuna over 60 networks. Selection maximised validation performance while minimising the train/validation MSE ratio. Then dropout and ℓ2 were raised by hand until TEST1/TEST2 passed.
- Finer meshes (smaller scale cut) overfit badly and cost too much memory.
- TARP works only on prior-distributed tests, so fixed-cosmology test sets use (μ − θ)/σ histograms instead.
- The authors admit they had no untouched final test set: they tuned against TEST1/TEST2.

## Caveats

- Robustness is bought by throwing away information (>2× precision, per 2310.15246).
- Regularisation was tuned on the test sets, so there is no truly blind test.
- Inconsistent H0 numbers.
