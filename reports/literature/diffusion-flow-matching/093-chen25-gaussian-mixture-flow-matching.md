# Gaussian Mixture Flow Matching Models

Hansheng Chen et al., 2025, ICML 2025 · [arXiv:2504.05304](https://arxiv.org/abs/2504.05304)

**Evidence: B.** Peer-reviewed (International Conference on Machine Learning); 20 citations in
1.5 yr (Semantic Scholar, 2026-10-19).

## What they do

- Predict a K-component Gaussian mixture over the random velocity u = (x_t − x0)/σ_t instead of its mean. Components share an isotropic std s.
- Train with NLL, i.e. the KL to p(u | x_t). This is a hybrid of centroid regression and component classification. K=1, s=1 recovers the L2 FM loss.
- Thm 3.1: at the optimum, the GM mean equals E[u] for any fixed covariances. The mean velocity is therefore the same as plain FM; only the extra distributional information is gained.
- The GM gives an analytic reverse transition, which yields GM-SDE/ODE solvers with sub-steps and 2nd-order variants.
- Probabilistic guidance reweights the GM with a Gaussian mask in place of CFG extrapolation.

## What they find

- 2-D checkerboard: GMFlow with K=64 is reasonable in 1 step and high quality in 4. Vanilla FM needs ~8 steps for a reasonable result and 16–32 for high quality.
- Larger K sharpens few-step samples. 2nd-order solvers matter only for small K or small NFE. Removing the q̂ conversion makes samples overshoot to the edges.
- ImageNet-256, DiT-XL/2 adapted to K=8 (+0.2% params):
  - Precision 0.942 at 6 steps with GM-ODE2; best precision 0.950 at 32 steps with GM-SDE2.
  - Saturation −0.019 versus +0.032 to +0.052 for the baselines (real data = 0.318).
- Validation NLL in bits/dim: 0.346 (K=1), 0.263 (K=4), 0.242 (K=8), 0.224 (K=16), 0.173 (K=16 + spectral).
- Metrics saturate at K=8.
- Ablations:
  - no spectral sampling: FID worse;
  - vanilla CFG on the GM: oversaturation;
  - DPM++ in place of GM-SDE2: worse;
  - KL loss in place of the transition loss (λ=0.5): FID worse;
  - no ODE sub-steps: much worse FID.
- Overhead: 0.005 s per step against 0.39 s.

## Implementation details

- Predict s with a tiny separate MLP on (t, c) only; this is more stable than predicting it from the trunk.
- Use the transition loss with Δt = λt to put a floor on the variance and avoid gradient spikes as s → 0.
- Factorise the mixture per pixel or per token.

## Caveats

- Gains are mainly in few-step sampling and guidance.
- Per-pixel factorisation loses correlations, which spectral sampling only patches.
