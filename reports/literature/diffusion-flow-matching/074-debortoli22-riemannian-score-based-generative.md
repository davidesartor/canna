# Riemannian Score-Based Generative Modelling

De Bortoli et al., 2022, NeurIPS 2022 · [arXiv:2202.02763](https://arxiv.org/abs/2202.02763)

**Evidence: A.** Peer-reviewed (Neural Information Processing Systems); 310 citations in 4.7 yr
(Semantic Scholar, 2026-10-19).

## What they do

Score SDEs on compact manifolds. The forward process is Brownian motion (or Langevin toward a wrapped target) on M. The reverse drift is ∇log p_t. Sampling is a geodesic random walk: an Euler–Maruyama step in T_xM followed by exp. Scores are trained with:
- DSM with a truncated eigen-expansion heat kernel,
- DSM with the Varadhan approximation t∇log p_{t|0} ≈ exp^{-1}_x(x0) (valid at small t), or
- (sliced) ISM.
The score is a tangent projection of an ambient network. There is a W1 convergence bound with e^{-λ1 T} mixing.

## What they find

- Earth NLL: RSGM −4.92 / −0.19 / 0.45 / −1.33, vs Moser −4.21 / −0.16 / 0.57 / −1.28. Stereographic SGM is worse (−3.80 / −0.19 / 0.59 / −1.28).
- SO(3) mixtures (M=16/32/64), LL: RSGM 0.89 / 0.20 / −0.49 at 0.1k NFE, vs Moser 0.85 / 0.17 / −0.49 at 2.3–7.3k NFE.
- RSGM is more robust than exp-wrapped SGM to the σ(t) hyperparameters (heatmap only).
- High-dim torus: Moser could not train for d ≥ 10 (figure only).

## Implementation details

- Use intrinsic geodesic random walks, not extrinsic projection, which accumulates error.
- The ODE likelihood is not the SDE-model likelihood (App. D).
- Varadhan DSM is cheap but biased at large t.

## Caveats

Low-dim only. No numeric loss ablation was found.
