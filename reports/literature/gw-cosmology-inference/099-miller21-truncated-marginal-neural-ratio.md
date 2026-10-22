# Truncated Marginal Neural Ratio Estimation

Miller et al., 2021, NeurIPS 2021 · [arXiv:2107.01214](https://arxiv.org/abs/2107.01214)

**Evidence: B.** Peer-reviewed (Neural Information Processing Systems); 62 citations in 5.3 yr
(Semantic Scholar, 2026-10-19).

## What they do

Train independent binary classifiers (BCE, joint vs shuffled pairs) for every 1-D and 2-D marginal likelihood-to-evidence ratio r(x|ϑ_k) on shared simulations (MNRE). Iterate rounds where the prior is truncated to a hyper-rectangle Γ: keep θ_d only where p̂(θ_d|x_o)/max p̂ > ε (≈ ±5.26σ for ε = 1e-6), simulate new data from the truncated prior, reuse old in-region data, stop when the prior-mass ratio between rounds > β = 0.8. The final estimator is *locally amortized* over Γ, so empirical HPDI coverage tests can be run on fresh simulations from the truncated prior.

## What they find

sbibm (all tasks except Bernoulli GLM, non-factorizable prior), metric C2ST-ddm (mean C2ST over all 1-D+2-D marginals): beats rej-ABC/SMC-ABC, beats non-sequential methods except on Gaussian Linear, competitive with sequential methods; best for narrow posteriors and large budgets, weak on wide ones (Gaussian Linear, SLCP). 5/240 runs failed to converge; 1–7 rounds. 3-D torus: stops after 4 rounds; TMNRE recovers the thin torus, MNRE without truncation misses its shape; ε scan 1e-10…1e-1 → 1e-6 optimal (most conservative value at the best metric). Early rounds are conservative (estimates converge from above), so truncation decisions are safe. 10-D eggbox (1024 modes), 10k sims: MNRE recovers structure, NRE/SNRE/SMNRE fail. CMB 6-param ΛCDM, 5k sims: MNRE matches converged MCMC (45k accepted samples) 1σ/2σ contours; NRE is barely constrained; a 5k-sample MCMC gets 1σ but not 2σ right.

## Implementation details

Resnet 2 blocks, 64 hidden, BN, batch 128, lr 1e-2 with plateau decay, early stopping patience 20 — early stopping credited with keeping estimates conservative. Shared linear compression of a 7497-dim data vector to 10 features feeding per-marginal MLP heads. Truncation error bound p_Γ = p + O(ε)·max p (App. C). Rectangle truncation is exact to sample (no rejection/SIR).

## Caveats

Marginals only (no joint, no posterior predictive) unless an extra joint estimator is trained. Box truncation is inefficient for correlated parameters (20% vs 99.94% rejection in the TSNPE comparison). Missing a mode in round 1 is unrecoverable and undetectable by the coverage test; locally amortized predictions are only trustworthy for θ well inside Γ (watch for contours hitting the box).
