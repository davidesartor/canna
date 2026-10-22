# Factorized neural posterior estimation for rapid and reliable inference of parameterized post-Einsteinian deviation parameters in gravitational waves

Zhang et al., 2026, Commun. Theor. Phys. 78, 115401; arXiv v2 Sep 2026 · [arXiv:2602.01319](https://arxiv.org/abs/2602.01319)

**Evidence: B.** Peer-reviewed (Communications in Theoretical Physics); 1 citation in 0.7 yr
(Semantic Scholar, 2026-10-19).

## What they do

replace one joint flow with 9 conditional 1-D flows q(δχ_i | x, θ*_i). The other ppE parameters are 0 in the simulations used to train estimator i. The conditioning parameters pass through an MLP "conditional embedding" shared with the data embedding.

## What they find

- Train: 2×10⁷ samples, IMRPhenomXPHM, H1+L1, 1 s at 4096 Hz.
- Test: 1000 injections; coverage passes the KS test (p > 0.05) for every parameter.
- Speed: 0.2–0.4 s per event vs 2.1–4.5×10⁴ s for dynesty (4096 live points), about 9×10⁴×.
- Training cost: 8–13 h per estimator on an A6000.
- Width vs MCMC: NPE 90% intervals are narrower than MCMC for δχ₀–δχ₂ and comparable for δχ₃–δχ₇ (Table 1). The authors attribute this to MCMC trapping in harmonic secondary modes, unverified.
- GW150914: m1 = 36.1 (+2.5, −3.7) vs 35.6 for LVK. D_L = 439 (+29, −40) vs 440 (+150, −170), suspiciously tight.

## Implementation details

- Data embedding: 3-layer 1-D CNN, then 128 residual blocks of 512 units with BN and dropout 0.1.
- Flow: 10 affine coupling layers, sub-nets of 5×512.
- Optimiser: AdamW, lr 1e-3, wd 1e-4, batch 2048, ≤200 epochs, early stopping with patience 15 and best-checkpoint restore.
- Normalisation: parameters scaled to [−1, 1]; strain z-scored per sample.
- Unknown conditioning parameters at inference: "sample from prior or iterative optimisation", i.e. Gibbs-like and not validated.

## Caveats

- Each estimator is a full conditional given the *true* other parameters, so the marginal posterior p(δχ_i | x) is not what is computed.
- The GW150914 run conditions on parameters from elsewhere, so its D_L width means nothing.
- Narrower-than-MCMC widths are claimed as a feature.
- Training sets each have one ppE nonzero, which is a different prior from the joint test.
- The calibration claims rest only on 1-D PIT/KS.
