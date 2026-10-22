# Dynamic(al) Conditional Optimal Transport through Simulation-Free Flows

Kerrigan, Migliorini, Smyth, 2024; NeurIPS 2024 · [arXiv:2404.04240](https://arxiv.org/abs/2404.04240)

**Evidence: B.** Peer-reviewed (Neural Information Processing Systems); 38 citations in 2.5 yr
(Semantic Scholar, 2026-10-19).

## What they do

conditional Wasserstein space is geodesic; triangular vector fields (zero y-velocity) generate its curves (conditional Benamou–Brenier). COT-FM: source η = P_Y ⊗ η_U (y resampled from the data y-marginal, u ~ Gaussian), minibatch OT with cost |y1−y0|² + ε|u1−u0|², ε≪1, then FM along straight lines in u with fixed y. Works in Hilbert spaces (FNO).

## What they find

2D (W2×1e-2): checkerboard 4.69 vs FM 8.81; moons 6.50 vs 15.55; circles 5.56 vs 7.03; swissroll 4.64 vs 8.18. Lotka–Volterra (4 params, vs DEMetropolis MCMC): W2 4.02 vs FM 11.41 (MMD 0.95 vs FM 2.65; COT-Flow best MMD 0.83). Darcy flow: MSE 5.40 vs FFM 7.30, CRPS 15.56 vs 15.47 (tie).

## Implementation details

ε searched in {1e-6,1e-4,1e-2,1e-1}; Darcy ε=1e-5; LV uses the *full* (not minibatch) COT plan over 10k pairs; 2D batches 256–1024; C=σ²I with σ∈{1e-3…0.5}.

## Caveats

no ablation over ε or batch size; y low-dimensional (1-D, 22 obs, or field with 60% noise); the large LV gain over FM is against a weak FM baseline tuned by random search.
