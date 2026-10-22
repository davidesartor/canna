# Variational Inference with Coverage Guarantees in Simulation-Based Inference (CANVI)

Patel, McNamara, Loper, Regier, Tewari, 2023; ICML 2024 · [arXiv:2305.14275](https://arxiv.org/abs/2305.14275)

**Evidence: B.** Peer-reviewed (International Conference on Machine Learning); 6 citations in
3.4 yr (Semantic Scholar, 2026-10-19).

## What they do

Split conformal prediction on top of any amortized posterior q.
- Score s(x,θ) = 1/q(θ|x). Set q̂ to the ⌈(N+1)(1−α)⌉/N empirical quantile on N fresh simulated (θ,x) pairs.
- The region C(x) = {θ : 1/q(θ|x) ≤ q̂}, a density-level set, so it can be disconnected.
- Guarantee: marginal coverage 1−α ≤ P(θ ∈ C(x)) ≤ 1−α + 1/(N+1) over the prior predictive, for any q.
- Model selection among candidate q's: minimize the expected Lebesgue volume of the region. Volume is estimated by importance-weighted Monte Carlo from a mixture of q and the prior over K mixing weights. A separate recalibration set restores exchangeability after selection.

## What they find

- ARCH model, coverage at 1−α = 0.5/0.75/0.9/0.95 before → after CANVI:
  - ELBO-trained: 0.0007/0.004/0.020/0.040 → 0.497/0.749/0.898/0.950.
  - IWBO-trained: 0.10/0.20/0.33/0.41 → 0.50/0.76/0.90/0.95.
  - FAVI-trained (i.e. NPE): 0.55/0.75/0.88/0.93 → 0.51/0.76/0.90/0.95.
- The FAVI/NPE-trained q gives the smallest corrected region (ℓ ≈ 0.75 vs 1.32 and 1.42).
- With K = 1 mixing weight the volume estimates blow up (19–70).
- On SBI benchmarks, calibration takes under 1 s per task using 10% of the budget.
- PROVABGS galaxy spectra (θ ∈ ℝ¹¹): all methods reach about 0.95 after correction. For ELBO, q̂(0.05) is about 0, so its region volume is infinite.

## Implementation details

Calibration sets can be large because simulation is cheap. Use about 10 mixing weights for the volume estimate. NPE-style (forward-KL) training gives the tightest corrected regions.

## Caveats

Marginal, not conditional, coverage; group-conditional coverage is only in the appendix. Requires the density q(θ|x). Volume estimation is noisy for under-dispersed q.
