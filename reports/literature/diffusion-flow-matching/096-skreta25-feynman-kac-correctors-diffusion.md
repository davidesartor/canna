# Feynman-Kac Correctors in Diffusion: Annealing, Guidance, and Product of Experts

Skreta, Akhound-Sadegh, ..., Doucet, Brekelmans, Tong, Neklyudov, 2025, ICML 2025 · [arXiv:2503.02819](https://arxiv.org/abs/2503.02819)

**Evidence: A.** Peer-reviewed (International Conference on Machine Learning); 109 citations in
1.6 yr (Semantic Scholar, 2026-10-19).

## What they do

Targets built from pretrained score models — annealed $q_t^\beta$, products $q_t^1q_t^2$, geometric averages (CFG), reward tilts $q_t e^{\beta_t r}$ — are not the marginals of the heuristic SDE that uses their summed/scaled score. Writing the target's PDE as continuity + diffusion + a *reweighting* term $\bar g_t p$ (Feynman–Kac) gives a weighted SDE: simulate the heuristic drift and accumulate $dw_t=g_t(x_t)dt$; Table 1 gives additive conversion rules. For linear forward drifts the weights for annealing/product/CFG need only score inner products and norms (e.g. product: $-\sigma_t^2\langle s^1,s^2\rangle$ plus drift-score cross terms), no divergences — zero extra network cost. SMC with systematic resampling on per-step increments; also a jump-process/birth–death formulation.

## What they find

EDM2-XS ImageNet-512 CFG: FKC lifts ImageReward from −0.24 to +0.04/+0.05 and CLIP 28.75→29.0–29.14 (benefit already at K=2, best K=8). LJ-13 Boltzmann: train DEM at T_L=2, anneal at inference to T_S=0.8 (β=2.5): target-score SDE distance-W2 0.189→0.048 with FKC, beating DEM trained directly at 0.8 (0.103); at T_S=1.5 tempered-noise+FKC best (energy-W1 0.77 vs DEM 4.47). Mixture of 40 Gaussians (exact scores): target-score+FKC best; tempered-noise SDE drops modes. Dual-target docking (100 protein pairs, PoE β=2): best max-docking −7.72 vs DualDiff −6.97, 45% better than reference binders, at a small diversity cost.

## Implementation details

resample only inside an "active interval" (e.g. τ∈[0,0.6]; t_max∈[0.5,0.7] best for molecules) — always-resampling cuts diversity, full-trajectory SNIS has too high variance. Final images in a batch often near-identical from weight degeneracy. Two simulation choices per β (target-score vs tempered-noise σ/√β), neither uniformly better. Time-dependent β_t needs log q_t, estimable with 2412.17762.

## Caveats

derivations assume the learned score is the exact score of q_t; with errors the weights correct to the wrong target and can add variance. Needs stochastic (SDE) sampling. No posterior-calibration experiments.
