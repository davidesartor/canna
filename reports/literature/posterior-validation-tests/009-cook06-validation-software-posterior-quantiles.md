# Validation of Software for Bayesian Models Using Posterior Quantiles

Cook, Gelman & Rubin, 2006, J. Comput. Graph. Stat. 15(3), 675–692 · [DOI:10.1198/106186006X136976](https://doi.org/10.1198/106186006X136976)

**Evidence: A.** Peer-reviewed (JCGS); 324 citations in 20.1 yr (Semantic Scholar,
2026-10-22).

## What they do

This is the origin of simulation-based calibration. Draw θ⁰ from the prior, simulate y from the model, then run the software under test on y to get L posterior draws θ¹…θᴸ. The posterior quantile of the truth is q = (1/L) Σₗ I(θ⁰ > θˡ). If the software is correct, q is uniform on (0, 1). Repeat for N replications.

Each scalar is tested with X² = Σᵢ (Φ⁻¹(qᵢ))², which is χ²_N under the null. They use the squared form because the mean of Φ⁻¹(q) misses U-shaped and ∩-shaped errors. Each scalar's p-value is turned into z = Φ⁻¹(p), the z values are plotted per batch of parameters, and a Bonferroni correction is applied across them.

## What they find

- Hierarchical normal model (J = 6 groups, 20 replications, L = 5000):
  - Correct software: all |z| < 2, and the smallest p-value is 0.2.
  - A bug using N instead of n_j: Bonferroni p ≈ 0.
  - A bug using 5 instead of 5² for a prior scale: Bonferroni p = 0.002.
- WinBUGS model:
  - A t₃ likelihood error gives p = 7.1 × 10⁻²⁹.
  - A hyperprior correlation error gives p = 8.2 × 10⁻¹⁵.
- Against Geweke's (2004) joint-distribution test on the same model, Geweke's method took about 4 min to converge (R̂ < 1.2) versus about 1 min for theirs (R̂ < 1.05). Geweke's method cannot be run inside WinBUGS at all, because it needs access to the sampler's transition.
- They state outright that a sampler that returns the prior passes their test, and Geweke's too. Quantiles of prior draws against a prior-drawn truth are uniform. For Geweke's that holds with parameter-only test functions; one coupling θ and y would catch it ([018](018-geweke04-getting-it-right-joint.md)).
- With only one replication, 15 of 16 p-values came out below 0.05 in one case. Several replications are needed.

## Implementation details

- Monitor functions of the parameters as well as the parameters themselves, for example cross-products. Errors in a correlation do not show in the marginals.
- Avoid "nice" fixed inputs such as round sample sizes or symmetric designs, which can hide indexing bugs.
- With a fixed budget, prefer fewer replications with enough posterior draws over many replications with small L.
- Use proper priors so that θ⁰ can be drawn.
- Quantiles need no finite prior variance. Geweke's moment tests do.
- Needs only posterior samples and a simulator. No density and no likelihood evaluation.

## Caveats

- Gelman (2017, JCGS correction) shows that the χ² test is not valid as stated: with finite L and autocorrelated MCMC draws, q is discrete and not exactly uniform, so the null distribution is wrong. Later work (Talts et al. 2018) moves to ranks of thinned draws.
- Tests are per scalar. Joint errors are visible only through the functions of θ the user chooses to monitor.
- The test checks the prior-averaged calibration. It cannot detect a sampler that returns the prior, and it does not localise errors to a region of data space.
- Bonferroni over many scalars is conservative and loses power.
