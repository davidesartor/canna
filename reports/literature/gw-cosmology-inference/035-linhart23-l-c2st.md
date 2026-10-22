# L-C2ST: Local Diagnostics for Posterior Approximations in Simulation-Based Inference

Linhart et al., 2023, NeurIPS 2023 · [arXiv:2306.03580](https://arxiv.org/abs/2306.03580)

**Evidence: B.** Peer-reviewed (Neural Information Processing Systems); 25 citations in 3.4 yr
(Semantic Scholar, 2026-10-19).

## What they do

- Train one amortised classifier d(θ, x) to separate class 0, (θ_q ~ q(θ|x), x), from class 1, (θ, x) ~ p(θ, x). Both classes use the same x_n.
- At any x_o, evaluate the statistic only on samples from q: t = mean over θ ~ q(·|x_o) of (d(θ, x_o) − ½)². It is zero iff q = p at x_o, for a Bayes-optimal classifier (Thm 3).
- Single-class accuracy is only a necessary condition; the MSE on predicted probabilities is the sufficient statistic.
- The null distribution comes from permutations.
- ℓ-C2ST-NF: pull the true θ back through the inverse flow and test z against N(0, I) jointly with x. Under the null, the classifiers can then be pre-trained once, independently of q, by drawing fresh Gaussians.

## What they find

- Two Moons and SLCP with N_cal = 10⁴: the statistic tracks the oracle C2ST.
- Type I error is controlled at α = 0.05 over 50 runs.
- At N_train = 10³:
  - ℓ-C2ST-NF reaches TPR = 1 at N_cal = 2000 on Two Moons.
  - Local-HPD never reaches TPR = 1 on SLCP.
  - The data-hungriest task (Bernoulli GLM, d = 10) needs N_cal ≈ 5000.
- Run time on SLCP is 2–44 s for ℓ-C2ST vs 280–960 s for local-HPD (at least 30× slower).
- JRNMM (4 parameters, 1024-sample time series, 10-layer MAF, N_train = 50k, N_cal = 10k):
  - Marginal SBC sees nothing wrong, and global HPD shows a mild global miscalibration.
  - ℓ-C2ST-NF shows a U-shaped failure in g_o: bad at the prior edges, fine near 0.
  - At g_o = 10 the colour-coded pairplot shows the posterior is over-dispersed and biased (+ for μ and σ, − for g).

## Implementation details

- Use an sklearn MLPClassifier with sbibm settings.
- Calibrate the classifier and check it by cross-validation, because the statistic uses the probabilities, not the labels.
- Colour the posterior pairplot by the predicted class probability. That shows *where* q has too much or too little mass, i.e. which parameter is biased or over-dispersed.
- Make local PP-plots per observation, with 95% null bands.
- Sweep x_o along one physical parameter (here g) to map where the amortised posterior breaks. In JRNMM it was at the prior edges.
- Hold out 𝒟_cal as data separate from training.

## Caveats

- Only as good as the classifier, and the classifier must ingest x as well. The paper's x are at most 10², or summary features; feeding a raw 4096-bin window to an MLP would cost power.
- Less sensitive than the oracle: the scatter points sit below the diagonal.
- Correlation with the oracle C2ST fades as N_train grows.
- The -NF variant needs an invertible map from θ to the base. For flow matching this means integrating the ODE backwards for every calibration pair, and it is not tested on FM or on manifold flows.
- Permutation nulls cost N_ℋ classifier trainings.
