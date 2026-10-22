# Solving high-dimensional parameter inference: marginal posterior densities & Moment Networks

Jeffrey, Wandelt, 2020, NeurIPS 2020 ML4PS workshop · [arXiv:2011.05991](https://arxiv.org/abs/2011.05991)

**Evidence: B.** Not peer-reviewed; 49 citations in 5.9 yr (Semantic Scholar, 2026-10-19).

## What they do

Two ways to skip the full joint posterior. (i) Marginal flows: train a conditional density estimator on only a subset (α,β) of θ (all of θ drawn from the prior), which automatically yields the marginal $p(\alpha,\beta|x)$. (ii) Moment Networks: a hierarchy of MSE regressors — $\mathcal F(x)$ minimising $\|\theta-\mathcal F(x)\|^2$ is the posterior mean; then with $\mathcal F$ frozen, $\mathcal G(x)$ minimising $\|(\theta-\mathcal F(x))^2-\mathcal G(x)\|^2$ is the posterior variance; analogous for covariances/higher moments.

## What they find

100-D linear-Gaussian-ish toy (non-stationary noise, correlated Gaussian prior): MAF-ensemble marginals and Moment Networks both trained on 8×10⁴ simulations match a 10⁷-sample MCMC reference (figure only). Moment Network trains in seconds, evaluates in O(10⁻²) s. LIGO-like toy (SEOBNRv4 + polarisation only, 35 000 sims, masses U(10,30), distance U(500,1500) Mpc): per-time-sample posterior σ of 128 strain values; validated against MCMC on a 64-D known-likelihood analogue (figure only).

## Implementation details

Use the disagreement between marginal flow moments and Moment Network moments as a cross-validation diagnostic (insufficient capacity or simulations). Moment networks cannot mode-collapse (they target moments by construction).

## Caveats

4-page workshop paper; no numbers beyond figures; no high-SNR or multimodal tests; moments are poor summaries for multimodal / circular posteriors.
