# Assessment and Adjustment of Approximate Inference Algorithms Using the Law of Total Variance

Yu, Nott, Tran & Klein, 2021, J. Comput. Graph. Stat. 30, 977–990 · [arXiv:1911.08725](https://arxiv.org/abs/1911.08725)

**Evidence: B.** Peer-reviewed (JCGS, DOI 10.1080/10618600.2021.1880921); 18 citations in
6.9 yr (Semantic Scholar, 2026-10-22).

## What they do

Two identities hold for any correct posterior:

- E(θ) = E[E(θ | Y)] (the tower property).
- Cov(θ) = E[Cov(θ | Y)] + Cov(E(θ | Y)) (the law of total variance).

The left sides come from prior draws. The right sides come from the approximate posterior's means and covariances over simulated datasets. A mismatch means the approximation is wrong. The check is multivariate by construction: it compares full mean vectors and covariance matrices, not marginals.

Estimators:

- μ_L, Σ_L from the prior draws θ⁽ⁱ⁾.
- μ_R as the average of the approximate posterior means, and Σ_R = Σ_R1 + Σ_R2: the average approximate posterior covariance plus the covariance of the approximate posterior means. Each replicate uses S posterior particles.

They bootstrap the replicate triples (B = 1000) and plot μ_L against μ_R, and the standard deviations and correlations from Σ_L against those from Σ_R. Points above the diagonal mean the approximation overestimates that quantity.

The same identities give an adjustment (eq. 9): θ̃ = μ_L + (μ_R(y) − μ_R) + T C⁻¹ (θ − μ_R(y)), with C = chol(Σ_R1) and T = chol(Σ_L − Σ_R2). If Σ_L − Σ_R2 is not positive definite, they shrink it by a factor ρ.

## What they find

- The prior used as the posterior satisfies both identities exactly. The authors say so, and follow Prangle et al. (2014): condition on simulated datasets Y in a set F near y_obs.
- With correct means, an under-dispersed approximation gives Σ_R < Σ_L, and an over-dispersed one gives Σ_R > Σ_L. So the check detects posteriors that are too wide or too narrow.
- Sum of log-normals (Rodrigues et al. 2018; Fenton-Wilkinson Laplace approximation; n = 10, κ = 10, (μ, σ) = (0, 1); priors N(0, 1) and Gamma(1, 1); 10,000 prior training pairs, of which F keeps 1,000): the approximation overestimates the μ means, underestimates the σ means and standard deviations, and overestimates the correlation. Here the quantile-based method of Rodrigues et al. adjusts best.
- Deep GLMM (German health-care panel, first 500 of 1887 workers, 12 predictors; approximate normal variational posterior of Tran et al. 2019; I = 200 prior replicates, 100 kept by a Jaccard-distance tolerance): the approximation underestimates the largest log Γⱼ and overestimates their variability. The total-variance adjustment cuts the squared estimation error substantially. The Rodrigues method does badly here, because its adjusted samples cannot leave the range of the original particles.
- Neural-network surrogate for Gaussian-process prediction (n = 1000, two hidden layers of 64): shows the method on predictive quantities.
- Their stated weakness: only first and second moments are checked and corrected. Errors in posterior shape are missed.

## Implementation details

- Needs only posterior samples (or the approximation's mean and covariance) and a simulator. No density and no likelihood.
- Cost is one approximate fit per kept replicate. With an amortised approximation this is cheap.
- The conditioning set F is a tolerance around y_obs on a chosen distance. Choosing ε so that a fixed number of replicates is kept worked in their examples.
- For dependent replicates (shared training sets in the GP example) the bootstrap plots need modification. They skip this.

## Caveats

- Moments only. Two different posteriors with the same mean and covariance per dataset pass identically.
- Without the conditioning set the check is blind to q = prior. With it, the check is local and depends on the tolerance.
- The bootstrap plots are a visual check. No formal test or power study is given.
- I read the arXiv v1 (2019). The published JCGS version may differ in details.
