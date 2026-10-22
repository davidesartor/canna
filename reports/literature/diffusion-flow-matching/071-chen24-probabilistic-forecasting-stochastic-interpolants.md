# Probabilistic Forecasting with Stochastic Interpolants and Föllmer Processes

Chen, Goldstein, Hua, Albergo, Boffi, Vanden-Eijnden, 2024; ICML 2024 · [arXiv:2403.13724](https://arxiv.org/abs/2403.13724)

**Evidence: B.** Peer-reviewed (International Conference on Machine Learning); 75 citations in
2.6 yr (Semantic Scholar, 2026-10-19).

## What they do

base is a point mass at the current state x0; I_s = α_s x0 + β_s x1 + σ_s W_s with σ_s=ε(1−s); SDE dX = b_s(X,x0)ds + σ_s dW with b learned by regression. Diffusion can be retuned post-hoc via b + ½(g²−σ²)∇log ρ (score from b); KL-optimal g^F gives a Föllmer process.

## What they find

Navier–Stokes: SDE vs deterministic mean-L² error 0.11 vs 0.23; long-time enstrophy error 5.6e-3 vs 0.30. KL(enstrophy) SDE σ_s 8.5e-3, Föllmer g^F 2.8e-3, Gaussian-base ODE 3.6e-3; KL(energy) 4.0e-3 / 7.2e-3 / 21.7e-3 — Föllmer helps one statistic, hurts the other. Video FVD KTH 39.1 vs RIVER (Gaussian-base FM) 41.9; CLEVRER 39.3 vs 49.0. >100× faster than SPDE.

## Implementation details

β_s = s² (β̇_0=0) gave more stable gradients and better spectra than β_s=s; σ_s=1−s best; 200 Euler–Maruyama steps.

## Caveats

a point-mass base *needs* the SDE (an ODE cannot spread a Dirac); improvements over Gaussian-base are moderate and statistic-dependent.
