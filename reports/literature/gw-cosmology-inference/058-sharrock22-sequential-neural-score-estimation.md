# Sequential Neural Score Estimation: Likelihood-Free Inference with Conditional Score Based Diffusion Models

Sharrock, Simons, Liu, Beaumont, 2022/2024, ICML 2024 · [arXiv:2210.04872](https://arxiv.org/abs/2210.04872)

**Evidence: B.** Peer-reviewed (International Conference on Machine Learning); 71 citations in
4.0 yr (Semantic Scholar, 2026-10-19).

## What they do

NPSE = conditional denoising score matching $\min\int\lambda_t\mathbb E\|s_\psi(\theta_t,x,t)-\nabla\log p_{t|0}(\theta_t|\theta_0)\|^2$ on prior–simulator pairs; sample by probability-flow ODE (RK45). Sequential variant TSNPSE: proposal = mixture of the prior truncated to the HPR$_\varepsilon$ of the previous round's posterior; because the truncated prior is ∝ prior on the posterior's support, no importance correction is needed (Prop. 3.1). SNPSE-A/B/C are score analogues of SNPE-A/B/C. Appendix D: multi-observation composition via $(1-n)\nabla\log p_t(\theta)+\sum_i s(\theta,x_i,t)$ with the *diffused* prior score (a variant of Geffner et al.).

## What they find

sbibm 8 tasks at 10³/10⁴/10⁵ sims, C2ST (figures only, no tables). NPSE/TSNPSE best or tied on the hardest tasks (SLCP, Lotka–Volterra); mixed elsewhere. Pyloric 31-param neuroscience model: 9 rounds (30k + 8×20k sims), 81% valid summary statistics in final round, higher than TSNPE and SNVI at equal budget. Ablation (App. C.5): TSNPSE ≫ SNPSE-A, SNPSE-B on SLCP and GLU; SNPSE-C "failed" (C2ST≈1) because the proposal-prior score must itself be learned. NPSE ≫ NLSE when the perturbed prior score has to be learned; ≈ equal when it is analytic. Conservative/energy parameterisation discussed only (App. G) to make HPR truncation cheap.

## Implementation details

VE SDE for low-dim, VP SDE for high-dim ("recommend"); VE σ_min=0.01 (2-D) / 0.05, σ_max by Song & Ermon "Technique 1"; VP β∈[0.1, 11]. Separate 3×256 MLP embeddings for θ_t and x (out dim max(30,4d)), 64-dim sinusoidal t embedding, concat → 3×256 SiLU MLP. Adam 1e-4, 15% held-out early stopping (patience 1000 steps), batch 50/200/500. Standardise θ and x. HPR via 20k ODE samples + exact log-density (Hutchinson-free trace) and ε=5×10⁻⁴ quantile; a cheap bounding-box pre-rejection cuts density evaluations. Same hyperparameters across all tasks (no tuning) — FMPE numbers taken from Dax et al. with per-task tuning, so the NPSE-vs-FMPE comparison (App. F) is not apples-to-apples.

## Caveats

low-dim benchmarks (≤10 params, except pyloric 31); no tables, only plots; computational cost of CNF density evaluation is the stated main limitation.
