# Fast Parameter Inference on Pulsar Timing Arrays with Normalizing Flows

Shih et al., 2023, PRL 2024 · [arXiv:2310.12209](https://arxiv.org/abs/2310.12209)

**Evidence: B.** Peer-reviewed (Physical Review Letters); 14 citations in 3.0 yr (Semantic
Scholar, 2026-10-19).

## What they do

Amortised NPE with a conditional MAF-RQS flow for p(θ|r), θ ∈ R^22 = (log10 A_GW, γ_GW, 10 × per-pulsar red-noise (log10 A_r, γ_r)), conditioned on 2,940 G-matrix-projected timing residuals of 10 NANOGrav-12.5yr pulsars (white noise fixed at 100 ns). Trained on 10^6 simulations (90/10 split) by maximum likelihood. Posterior then optionally importance-reweighted with the exact Gaussian PTA likelihood, w = L·π/q, evidence = mean(w).

## What they find

Raw-flow vs MCMC Hellinger distance over 10 realisations (Gaussian approx. to the 2-D GWB marginal): 0.33 ± 0.04; after likelihood reweighting 0.22 ± 0.15. IS efficiency ε = (Σw)²/(NΣw²): log10 ε = −3.2 ± 0.7 with N = 10^6 (i.e. ~0.1%), vs −5.7 ± 0.2 using the prior as proposal (ESS ≈ 1). Evidence from flow-IS is 10^9.7 larger than prior-IS (prior-IS is useless). Training ~24 h on 3 P100s; sampling in seconds.

## Implementation details

(i) Inputs spanned ~14 orders of magnitude and both signs: fixed rescale r → 10^7 r then clip at ±1000 to make inputs O(1) — deliberately sacrifices the loud-signal regime. (ii) θ affinely mapped to [−1,1]^22; base distribution *uniform* on [−1,1] with RQS tail bound 1, reported better than a Gaussian base for a uniform prior. (iii) Embedding: per-pulsar LSTM (hidden 100, 2 layers, outputs final hidden+cell) → MLP (200,100 → 50) per pulsar, then concatenation → MLP (100,100 → 50). LSTM per pulsar "improved performance significantly" over an MLP, and the two-stage per-pulsar structure beat a single LSTM/MLP over all residuals (no numbers given). (iv) 8 MADE blocks, 2×200 hidden, 8 spline bins; RAdam default; ≤100 epochs, best-validation epoch kept. (v) Timing-model marginalisation by G-matrix projection of the inputs (data-side analytic marginalisation).

## Caveats

Only qualitative ablations (no numbers). Hellinger computed under a Gaussian approximation. ε ~ 10^-3 means IS is expensive (10^6 likelihood calls) even in 22-D — raw flow was not tight. Simplified simulator (white noise fixed, no DM noise). No coverage/SBC test.
