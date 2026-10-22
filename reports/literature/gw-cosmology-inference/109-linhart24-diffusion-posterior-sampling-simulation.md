# Diffusion posterior sampling for simulation-based inference in tall data settings

Linhart, Cardoso, Gramfort, Le Corff, Rodrigues, 2024, arXiv; later TMLR-style · [arXiv:2404.07593](https://arxiv.org/abs/2404.07593)

**Evidence: B.** Peer-reviewed (Trans. Mach. Learn. Res.); 20 citations in 2.5 yr (Semantic
Scholar, 2026-10-19).

## What they do

Tall data = n observations i.i.d. given θ; a single-observation score network $s_\phi(\theta_t,x,t)$ is trained (NPSE) and composed at sampling time. The exact tall posterior diffused score is $(1-n)\nabla\log p^\lambda_t(\theta) + \sum_j\nabla\log p_t(\theta|x_j) + \nabla\log L_\lambda(\theta_t,x_{1:n})$; the last correction term is what F-NPSE (2209.14249) drops and why it needs Langevin correctors. Approximating the backward kernels $p_{0|t}$ as Gaussian (Tweedie second order) gives a closed-form precision-weighted sum: $\Lambda^{-1}\big(\sum_j\Sigma_{t,j}^{-1}s_j + (1-n)\Sigma_{t,\lambda}^{-1}s_\lambda\big)$, $\Lambda=\sum_j\Sigma_{t,j}^{-1}+(1-n)\Sigma_{t,\lambda}^{-1}$, usable with a deterministic DDIM sampler. Two Σ estimates: JAC (Jacobian of the score network, Tweedie) and GAUSS ($\Sigma_t=(\Sigma_0^{-1}+\alpha_t/\upsilon_t I)^{-1}$ with $\Sigma_0$ the empirical covariance of ~100 single-observation DDIM posterior samples per $x_j$).

## What they find

Gaussian toy (m=10, n=32, ε=1e-2), sliced Wasserstein: GAUSS 0.17–0.22 across T=50–1000 steps; JAC 3.14 (T=50) → 0.25 (T=1000); Langevin NaN at T=50, ~0.65 thereafter. Wall-time relative to Langevin: GAUSS 0.37–0.39× (≈2.5× faster), JAC 0.52–0.56×, flat in m; equal-time comparison (GAUSS T=1000 vs JAC/Langevin T=400) still favours GAUSS. With learned (noisy) scores on toy Gaussian GAUSS is best; on a GMM toy it beats Langevin for small n and ties for n>90. SBI benchmarks (SLCP, SIR, Lotka–Volterra; N_train 1e3–3e4; n up to 30; 25 seeds): GAUSS best overall; Langevin diverges for n≥14 on LV and SIR; JAC-clip ≈ GAUSS, slightly better on SLCP. Error grows with n for all (score errors add); PF-NPSE-style partial factorisation mitigates. JRNMM neural mass model (50k sims): only GAUSS passes ℓ-C2ST in both 3-D and 4-D; 4-D tall posterior reveals g–(μ,σ) degeneracy.

## Implementation details

JAC unstable with noisy scores, diverges for n>1 unless the Jacobian-derived covariances are clipped (JAC-clip). Langevin baseline (L=5 corrector steps, τ=0.5) is step-size sensitive; tamed ULA helps but still diverges (App. K). Geffner's code unavailable — Langevin results are the authors' reimplementation. The prior score term with weight (1−n) needs an analytic or separately learned diffused prior score.

## Caveats

Requires conditionally i.i.d. observations; Gaussian kernel approximation is poor for multimodal posteriors (GMM case only ties). Low-dim θ (≤10). No flow-matching (velocity) variant, though the same algebra applies to Gaussian-path FM via velocity↔score conversion.
