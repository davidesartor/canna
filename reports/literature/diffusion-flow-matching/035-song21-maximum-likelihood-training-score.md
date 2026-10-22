# Maximum Likelihood Training of Score-Based Diffusion Models

Song, Durkan, Murray, Ermon, 2021, NeurIPS 2021 · [arXiv:2101.09258](https://arxiv.org/abs/2101.09258)

**Evidence: A.** Peer-reviewed (Neural Information Processing Systems); 967 citations in 5.7 yr
(Semantic Scholar, 2026-10-19).

## What they do

The weighted score-matching objective J_SM(θ, λ) with the likelihood weighting λ(t) = g(t)² (the SDE diffusion coefficient squared) upper-bounds KL(p‖p_θ^SDE) − KL(p_T‖π) (Thm 1, via the data-processing inequality on path measures plus Girsanov).
- Thm 2: the bound is tight when s_θ is the exact score of some diffusion started at π; then p^SDE = p^ODE.
- Thm 3 gives a per-datapoint bound, the continuous-time ELBO.
- The bound is for the SDE sampler. The probability-flow ODE (a CNF) is not guaranteed to improve, but empirically −log p^ODE ≤ bound in every setting.

## What they find

Table 2, DDPM++ models, CIFAR-10 NLL in bits/dim with uniform dequantisation, and FID:
| Model | NLL | FID |
|---|---|---|
| Baseline VP | 3.16 | 3.98 |
| + likelihood weighting | 3.06 | 5.18 |
| + likelihood weighting + importance sampling | 2.95 | 6.03 |
| Deep VP | 3.13 | 3.09 |
| Deep + LW + IS | 2.93 | 5.34 |
- Likelihood weighting alone improves CIFAR-10 but not ImageNet32 (VP baseline 3.90 → 3.91). Importance sampling is needed for uniform gains (ImageNet32 3.86).
- Loss variance after convergence with importance sampling: CIFAR-10 98.48 → 0.068; ImageNet 0.51 → 0.043.
- ScoreFlow (LW + IS + variational dequantisation): 2.83 bits/dim on CIFAR-10, 3.76 on ImageNet32.
- Likelihood comes at an FID cost: CIFAR-10 2.86 → 5.34; ImageNet32 8.34 → 10.18.

## Implementation details

- Importance sampling: the time integral ∫g²h dt = Z∫p(t)α²h dt with p(t) = g²/(α²Z) (Eq. 12), sampled by inverse CDF. No loss history is needed, unlike the IDDPM approach.
- Start time ϵ = 1e-5 (VP), 1e-2 (subVP; 1e-5 made the ODE solver stiff under likelihood weighting).
- Test-time bounds use a Tweedie-based Gaussian denoiser q(x|x′) for the last step to [0, ϵ].
- About 550 (CIFAR-10) and 450 (ImageNet32) network evaluations for ODE sampling and likelihood.

## Caveats

- Images only, no learned schedule.
- The guarantee is for the SDE, not the ODE that is usually sampled.
- Likelihood weighting is heavy-tailed in t, so the raw estimator has very high variance.
