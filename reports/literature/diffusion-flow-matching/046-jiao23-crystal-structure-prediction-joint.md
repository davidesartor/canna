# Crystal Structure Prediction by Joint Equivariant Diffusion

DiffCSP; Jiao et al., 2023, NeurIPS 2023 · [arXiv:2309.04475](https://arxiv.org/abs/2309.04475)

**Evidence: A.** Peer-reviewed (Neural Information Processing Systems); 249 citations in 3.2 yr
(Semantic Scholar, 2026-10-19).

## What they do

Joint diffusion of the lattice L (Euclidean DDPM, cosine schedule) and fractional coordinates F on the torus. F uses wrapped-normal score matching with exponential σ (σ1=0.005, σ_T=0.5), T=1000, uniform prior, and a predictor–Langevin corrector. Periodic translation invariance comes from a Fourier transform of relative fractional displacements, sin/cos(2πm·Δf) with 256 dims.

## What they find

- MP-20 CSP match rate 51.49% (1 sample) / 77.93% (20), vs CDVAE 33.90 / 66.95.
- Ablations on MP-20 (match % / RMSE), full model 51.49 / 0.0631:
  - **w/o wrapped normal (Gaussian DDPM on F): 34.09 / 0.2350.**
  - **w/o Fourier features (raw relative F): 29.15 / 0.0926** (overfits, invariance broken).
  - w/o O(3)-invariant L input: 1.66 / 0.4002.
  - two-stage F→L: 36.73.
- σ_max ablation: **σ_max=0.1 → 32.56%** (the wrapped normal is not ≈ uniform, so a prior mismatch), σ_max=1.0 → 47.89%, default 0.5 → 51.49%.

## Implementation details

- Wrapped-normal score from a lattice sum truncated at |k|≤10 (21 images).
- λ_t = 1/E‖score‖², estimated with 10k MC samples.
- The Langevin step size must be tuned per dataset (5e-7 to 1e-5).

## Caveats

Diffusion only, with a strong symmetry prior. Match rate is a coarse metric.
