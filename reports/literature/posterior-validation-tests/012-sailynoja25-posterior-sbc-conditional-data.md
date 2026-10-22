# Posterior SBC: Simulation-Based Calibration Checking Conditional on Data

Säilynoja, Schmitt, Bürkner & Vehtari, 2025, Statistics and Computing (2026) · [arXiv:2502.03279](https://arxiv.org/abs/2502.03279)

**Evidence: B.** Peer-reviewed (Statistics and Computing, DOI 10.1007/s11222-026-10825-9); 13
citations in 1.7 yr (Semantic Scholar, 2026-10-22).

## What they do

Prior SBC averages calibration over the whole prior predictive. A method can be well calibrated on average and still fail near the observed data. Posterior SBC moves the check to the region the observed data selects.

The procedure for each iteration:

1. Draw θ′ from the fitted posterior q(θ | y_obs).
2. Simulate new data y ~ p(y | θ′).
3. Fit the augmented posterior q(θ | y, y_obs), conditioning on both datasets, and take thinned draws θ″.
4. Compute the PIT of θ′ among the θ″ (their eq. 3).

If both fits are correct, θ′ is a draw from the augmented posterior's own prior-predictive. The PIT is then uniform. A failure of either the original or the augmented fit breaks uniformity. No draws from the true posterior are needed.

## What they find

- Fig. 1 argument: biases that cancel over the prior are hidden in prior SBC and exposed in posterior SBC. Conversely, a failure in a region far from y_obs flags prior SBC but not posterior SBC.
- Hierarchical model (50 groups × 5 observations; μ₀ ~ N(0, 1); σ, τ ~ N⁺(0, 1)):
  - Two datasets: τ = 0.06, σ = 1.96 and τ = 1.96, σ = 0.06 (the 5th and 95th prior percentiles), μ₀ = 0.
  - Stan NUTS, adapt_delta = 0.99, 4 chains × (1000 warm-up + 1000 draws).
  - Prior SBC with 500 iterations found no problem for either parameterisation, although 12% of centred fits had divergences.
  - Posterior SBC with 500 iterations: the non-centred version fails on μ₀ and τ (thin right tail) for the strongly informative dataset. The centred version overestimates τ for the weakly informative one.
- Lotka-Volterra (hare/lynx data 1900–1920; α, γ ~ N(1, 0.5); β, δ ~ N(0.05, 0.05); log σ ~ N(−1, 1); Pathfinder initialisation; bimodal posterior):
  - Cost: 250 prior-SBC iterations took 6.5 h. 500 posterior-SBC iterations took 2.5 h, over 5× faster per iteration, because fits near y_obs are easier.
  - Prior SBC flags the log-likelihood (too few PIT values in 0.5–1). Posterior SBC is fine near the observed data.
- Amortised drift-diffusion models (BayesFlow-style: DeepSet summary with a 32-dim output, 6 affine coupling layers; N_obs ~ U(50, 150); 300 epochs, batch 64, 1000 iterations per epoch, cosine learning rate 5 × 10⁻⁴; two models fitted to real data):
  - Prior SBC at both N_obs and 2·N_obs looks fine.
  - Posterior SBC shows minor problems for one model on one dataset, and a pathological μ(e) for the other model on both datasets.

## Implementation details

- Use the joint log-likelihood as a test quantity alongside the parameters, as in Modrák et al. (2023).
- The model must accept the doubled dataset (y_obs plus y). For an amortised network this means it must be trained on, or at least accept, twice the data size.
- To keep the augmented posterior close to the original, the authors suggest conditioning on part of the observed data first.
- PIT values are shown with 95% simultaneous ECDF bands from `bayesplot` (Säilynoja et al. 2022).
- Needs only posterior samples and a simulator. No density or likelihood is needed for the parameter checks; the log-likelihood quantity needs the likelihood.
- Code: github.com/TeemuSailynoja/posterior-sbc.

## Caveats

- Passing is necessary, not sufficient, like prior SBC.
- The check concerns the augmented posterior, which is narrower than the one actually used. A fit can be calibrated on 2N observations and not on N, or the reverse.
- The number of posterior-SBC iterations in the amortised case is not stated in the main text.
- Whether q = prior is caught is not discussed directly. In this scheme θ′ would be a prior draw and θ″ also prior draws, so the parameter PITs would be uniform; only data-dependent quantities could catch it. This is my reading, not a claim of the paper.
