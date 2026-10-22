# Validating Bayesian Inference Algorithms with Simulation-Based Calibration

Talts et al., 2018, arXiv preprint · [arXiv:1804.06788](https://arxiv.org/abs/1804.06788)

**Evidence: A.** Not peer-reviewed; 448 citations in 8.5 yr (Semantic Scholar, 2026-10-19).

## What they do

Draw θ̃ ~ prior, ỹ ~ p(y|θ̃), then L posterior draws θ_1..L ~ p̂(θ|ỹ). For any 1-D statistic f, the rank of f(θ̃) among {f(θ_l)} is uniform on {0..L} iff the algorithm is computationally faithful (self-consistency of the data-averaged posterior with the prior). Algorithm 1 handles independent samples. Algorithm 2 handles correlated MCMC: it thins to N_eff draws, because autocorrelation alone produces end spikes.

## What they find

- Default setup: N = 10,000 replications, L = 100. The gray band is the 99% interval of Binomial(N, 1/(L+1)) per bin. Rebin to B bins so N/B ≈ 20, and pick L+1 divisible by the bin count (e.g. L = 1023).
- Diagnostic shapes: ∩ means over-dispersed (too wide), ∪ means under-dispersed (overconfident), and a slope or asymmetry means bias.
- Case studies:
  - A misspecified (narrower) prior gives a clear ∪.
  - Centred eight-schools HMC is biased in τ: there are too few small-τ draws.
  - Mean-field ADVI on linear regression is strongly biased on β.
  - INLA bias is only caught by the ECDF-difference plot at N = 1000.
- χ² uniformity tests had poor power and are not recommended over visual histograms and ECDFs.

## Implementation details

- Thin MCMC chains to N_eff before ranking.
- Use several test functions f, e.g. each parameter plus the log-likelihood.
- Prefer the ECDF-difference plot for subtle deviations.
- The prior used to simulate must match the prior of the fit.

## Caveats

- SBC only checks 1-D projections, averaged over data.
- A prior-returning approximation p̂(θ|y) = p(θ) passes exactly.
- It says nothing about any single observation.
- Ranks of an angle on a circle depend on where the cut is placed.
