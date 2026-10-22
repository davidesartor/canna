# Discriminative Calibration: Check Bayesian Computation from Simulations and Flexible Classifier

Yao & Domke, 2023, NeurIPS 2023 · [arXiv:2305.14593](https://arxiv.org/abs/2305.14593)

**Evidence: B.** Peer-reviewed (Neural Information Processing Systems); 2 citations on its
Semantic Scholar record in 3.4 yr (Semantic Scholar, 2026-10-22). That count looks like a split
record and is likely an undercount.

## What they do

SBC is recast as a classification problem. Simulate S pairs (θ, y) ~ p(θ)p(y|θ). For each y, draw M samples θ̃₁…θ̃_M ~ q(·|y). Turn this table into labelled examples (t, ϕ) and train any classifier to predict t. The classifier's held-out log predictive density, minus the label entropy, is a divergence estimate D. D is always a lower bound on a true divergence between p(θ|y) and q(θ|y), averaged over y. A permutation test on the same classifier gives an exact p-value.

The paper gives four label mappings:

1. **Binary with y.** The true θ gets t = 0 and each θ̃ gets t = 1, with features (θ, y).
2. **Binary without y.** The same, with features θ only.
3. **Binary on ranks.** Features are the rank of θ among the θ̃. This recovers SBC.
4. **Multiclass.** Shuffle θ among the M + 1 draws. The label t is the position of θ, and K = M + 1 classes.

## What they find

- **Ordering.** The divergences satisfy D4 ≥ D1 ≥ D3 ≥ D2. The multiclass mapping is the most powerful.
- **Limit.** As the classifier becomes optimal, D4 → KL(p‖q) − χ²(q‖p)/(2M).
- **Blind prior.** If q(θ|y) = p(θ), the rank divergence D3 is 0. The authors call this "a known counterexample of when rank-based SBC fails". D1 and D4 use y as a feature, so in principle they can detect it.
- **Gaussian null** (d = 16, S = 500, true posterior as q, 1000 repeats): the permutation p-values are uniform.
- **Gaussian power** (5% level, 1000 repeats):
  - Bias corruption: a scalar of 0.01 to 0.2 added to each posterior-mean coordinate.
  - Width corruption: the posterior covariance multiplied by 0.8 to 1.2. The 1.2 case is a too-wide posterior.
  - The binary classifier has uniformly higher power than Bonferroni-corrected χ² SBC. It is "sometimes as good as SBC with a 10 times bigger simulation sample size".
- **MCMC.** On three SBI-benchmark models (SLCP, Gaussian linear, Gaussian mixture), NUTS is stopped after 2 to 2000 iterations. The estimated JS divergence falls toward 0 as the iterations grow, and flaws are detected at early iterations.
- **Galaxy clustering.** θ is 14-D. Three observation designs have d_y = 38, 151 and 1031. Each fit uses S = 2000 and M = 500, which is about 1M examples. Estimated JS (sd):

  | q | d_y = 38 | d_y = 151 | d_y = 1031 |
  |---|---|---|---|
  | Best single flow | 0.015 (0.004) | 0.076 (0.007) | 0.068 (0.007) |
  | 5-flow ensemble | 0.013 (0.004) | 0.044 (0.006) | 0.035 (0.006) |

  These numbers use log q(θ|y) as a feature. A black-box MLP on the raw (θ, y) vector "would pass a t-test and not even detect miscalculation".
- **Visual check.** Plot the binary classifier's probability against one parameter. A U or inverted-U trend reveals under- or over-confidence in that parameter (their Fig. 6 shows covariance scaled by 0.8, 1 and 1.2). In the cosmology case, the plot found an under-confident concentration parameter, more clearly in the joint than in the marginal. The log odds also fall with log q(θ|y), which the authors read as q > p in q's tails, more evidence that q is underconfident.

## Implementation details

- **Classifier.** An MLP, with "linear features" added to the final logit: inv_logit[MLP(θ, y) + wᵀ l(θ, y)]. Here l holds log p(y|θ), log q(θ|y) and log p(θ), whichever are available. For the multiclass case, use a symmetric g(θ_k, y) with a softmax over the M + 1 draws.
- **Without a density.** When log q is unknown, use a rank feature: the fraction of the M draws below θ, which approximates the CDF of q. This needs only samples. The paper gives no result on how much power is lost without log q.
- **Train/validation split.** Split by simulation batch, never by individual example.
- **Permutation test.** Train the classifier once. Then permute labels within each simulation batch B times and recompute the validation LPD. The p-value is the fraction of permutations at or above the observed LPD. This is exact for any S, M or classifier. Permuting across batches is not valid. B is not fixed in the text.
- **Class imbalance.** With binary labels and large M, D1 → 0. Reweight the classes by C = (M + 1)²/(2M), which recovers the conditional JS divergence.
- **MCMC draws** need no thinning. The theorems hold even with M = 1.
- **Sufficient summary.** The trained classifier's output is a 1-D statistic that can itself be fed to a KS or permutation test.
- **Cost.** The 1M-example cosmology fit took about 2 CPU-hours on a laptop. The authors note that simulation dominates the total cost.
- **Code.** JAX: github.com/yao-yl/DiscCalibration.

## Caveats

- A global test, averaged over y. It says nothing about one particular y_obs.
- D is a lower bound. A weak classifier gives low power, and the cosmology example shows how much depends on the hand-built features.
- The high-dimensional y case worked with known log q and the flow's own features. Results with sample-only q and raw high-dimensional y are not reported.
- Blind-prior detection by D1/D4 is a consequence of the theory. It is not run as a separate experiment.
