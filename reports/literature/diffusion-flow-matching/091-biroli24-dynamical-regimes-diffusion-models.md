# Dynamical Regimes of Diffusion Models

Biroli et al., 2024, Nature Communications 2024 · [arXiv:2402.18491](https://arxiv.org/abs/2402.18491)

**Evidence: A.** Peer-reviewed (Nature Communications); 193 citations in 2.6 yr (Semantic
Scholar, 2026-10-19).

## What they do

The setting is the exact empirical score with large n and d. The backward OU dynamics has three regimes:
- I: Brownian.
- II: after "speciation" at t_S, trajectories have committed to a class. This is symmetry breaking, a Curie–Weiss double-well in q = m·x/√d.
- III: after "collapse" at t_C, trajectories are attracted to a single training point. This is a glass transition, computed with a Random Energy Model.

## What they find

- Speciation criterion: Λ e^{-2t_S} = 1, where Λ is the top eigenvalue of the data covariance. This is "class separation along the top PC equals the noise variance." For a GMM with |m|² = d μ̃², t_S = ½ log d.
- The cloning probability φ(t) (Eq. 6) collapses for d up to large values. Crossing is at φ ≈ 0.775.
- Collapse: t_C = ½ log(1 + σ²/(n^{2/d}-1)). Avoiding memorisation needs log n/d = O(1), i.e. exponentially many samples (the curse of dimensionality).
- Images, two-class subsets, U-Net, T = 5.05 (Table 1):
  | Dataset | n | d | Λ | t_S |
  |---|---|---|---|---|
  | MNIST | 10k | 1024 | 7.66 | 1.02 |
  | CIFAR | 3k | 3072 | 16.72 | 1.41 |
  | ImageNet16 | 2k | 768 | 3.05 | 0.56 |
  | ImageNet32 | 2k | 3072 | 12.11 | 1.25 |
  | LSUN | 40k | 12288 | 60.52 | 2.05 |
- The φ(t)/t_S curves collapse across datasets. The collapse time, estimated from the cloning and last-nearest-neighbour-change methods, matches the excess-entropy criterion with n = 200–2000.

## Implementation details

The speciation time is computable from the data covariance alone: it is when the noise variance equals the between-mode variance along the separating direction. That is a recipe for where to place time-sampling density for a given mode separation. The collapse time is diagnosed by an excess-entropy estimate.

## Caveats

The analysis assumes an exact empirical score, unconditional generation, a Gaussian base and OU noise. Speciation is about which class is selected by noise. The theory is physics-level rigour.
