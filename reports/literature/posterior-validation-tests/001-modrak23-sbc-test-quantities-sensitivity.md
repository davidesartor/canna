# Simulation-Based Calibration Checking for Bayesian Computation: The Choice of Test Quantities Shapes Sensitivity

Modrák, Moon, Kim, Bürkner, Huurre, Faltejsková, Gelman & Vehtari, 2023, Bayesian Analysis · [arXiv:2211.02383](https://arxiv.org/abs/2211.02383)

**Evidence: B.** Peer-reviewed (Bayesian Analysis, DOI 10.1214/23-BA1404); 73 citations in
4.0 yr (Semantic Scholar, 2026-10-22).

## What they do

SBC usually ranks each parameter's prior draw among its posterior draws. The authors generalise this to any test quantity f(θ, y), which may depend on the data. Ties are broken at random. They then ask which f make SBC sensitive to which failures, and prove what SBC can and cannot detect.

The main theory results:

- **Theorem 5.** Ranks of f are uniform on every subset of data space if and only if the implied posterior of f is correct.
- **Theorem 6.** If f is the density ratio π_post(θ | y) / φ(θ | y) between the true and the tested posterior, SBC with that single f is complete: passing implies φ is correct. Not computable in practice, but it motivates likelihood-based quantities.
- **Theorem 7.** If φ is the prior, SBC passes for every data-independent f. Only data-dependent quantities can catch it.
- **Theorem 8.** Monotone transforms of f give equivalent checks.

They also show that SBC and the data-averaged-posterior check (Geweke) are different conditions. In their Example 2, each can pass while the other fails. SBC constrains the quantile function, and the data-averaged posterior constrains the density. A convex combination of posteriors keeps the data-averaged posterior but breaks SBC. Cockayne et al. (2022, Lemma 2.19) show that SBC passing for every data-independent f implies a correct data-averaged posterior.

They summarise uniformity with the γ statistic (their eq. 7): the smallest two-sided binomial tail probability of the rank ECDF over all points. They report log(γ / γ̄), where γ̄ is the 5th percentile of γ under the null. Negative values mean a failure.

## What they find

- Multivariate normal model (correlation 0.8, n = 3 observations):
  - Posterior = prior: the joint log-likelihood fails within a handful of simulations. Every parameter-only quantity passes.
  - Ignoring 1 of 3 data points: about 20 simulations to detect.
  - Ignoring 1 of 20 data points: the joint log-likelihood is still weak after 1000 simulations. The pointwise log-likelihood of each observation detects it quickly.
  - Wrong posterior correlation: the parameters pass. Their sums, differences and products fail, and so does the log-likelihood.
  - A posterior built to compensate errors across parameters needs |μ| to show.
- Ordered-simplex model with an off-by-one Jacobian: log prior and x[1] catch it in about 400 simulations. The log-likelihood needs about 3000.
- γ is more sensitive than KS or χ² rank tests in their comparisons.

## Implementation details

- Default set of test quantities: all parameters, the joint log-likelihood, the log prior, and a few products or differences of correlated parameters.
- The likelihood quantity needs log p(y | θ) evaluated at each posterior draw. It does not need the posterior density.
- Use a number of SBC simulations S of the same order as the number of posterior draws M per simulation.
- Thin to at least 100 effective samples per fit, so the ranks are not inflated by autocorrelation.
- Code: the `SBC` R package; paper scripts at github.com/martinmodrak/sbc_test_quantities_paper.

## Caveats

- SBC is necessary, not sufficient. Passing any finite set of quantities does not prove the posterior correct.
- The complete test (Theorem 6) needs the true posterior density, which is unavailable.
- Power to detect localised problems is low, because ranks are averaged over the whole prior predictive.
- The examples are small Stan models. No high-dimensional or amortised examples.
