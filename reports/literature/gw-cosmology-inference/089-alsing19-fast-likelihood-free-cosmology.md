# Fast likelihood-free cosmology with neural density estimators and active learning

Alsing, Charnock, Feeney, Wandelt, 2019, MNRAS 488, 4440 · [arXiv:1903.00007](https://arxiv.org/abs/1903.00007)

**Evidence: A.** Peer-reviewed (Monthly notices of the Royal Astronomical Society); 223
citations in 7.6 yr (Semantic Scholar, 2026-10-19).

## What they do

pydelfi: learn the *likelihood* p(t|θ) of compressed summaries with an ensemble of NDEs (MDNs with 1–5 Gaussian components + a MAF of 5 MADEs), stacked with weights from validation loss; acquire simulations with SNL using proposal q = √(p̃(θ|t_o) π(θ)) (geometric mean of current posterior and prior, to cover tails) or with Bayesian-optimisation acquisition (ensemble variance). Reviews score compression, nuisance hardening t̄_θ = t_θ − F_θη F_ηη⁻¹ t_η, and IMNNs.

## What they find

JLA (6 params): converged after O(10³) sims vs 20k in 1801.01497 (GMM joint) and vs 6000 for BOLFI on only 2 params; matches MCMC. Euclid-like tomographic pseudo-C_ℓ cosmic shear (5 params; 5 z-bins, 10 ℓ-bands, MOPED-compressed): converged after O(10³) sims. Lyα forest Γ_HI (1 param; 50 flux percentiles → IMNN with 128/64/32 leaky-ReLU): converged after O(10²) sims. No ablation separating the effect of active learning, ensembling or Fisher pretraining.

## Implementation details

Argue for NLE (option 3) as default: parameter proposal is free (no importance reweighting of the posterior loss), compressed summaries are near-Gaussian so p(t|θ) is easy. **Fisher pre-training:** cast score summaries as pseudo-MLEs t → θ_* + F⁻¹t, then pre-train the NDE on 10⁶ synthetic pairs from 𝒩(t | θ, F⁻¹) with θ ~ prior; discard and fine-tune on real sims. Adam lr 1e-3, batch = 1/10 of training set, 10% validation, early stopping patience 20; tanh activations; 2×50 hidden. Compression mean/cov from 10³ sims, derivatives from 100 seed-matched pairs per parameter at ±5% steps. Recommends compressing to physically trusted first-level summaries first, because deep compression of raw data can latch onto simulator artefacts.

## Caveats

No quantitative ablations; convergence judged by eye/inset curves; low dimensional (≤6 params); "matches MCMC" only on JLA where compression is exact.
