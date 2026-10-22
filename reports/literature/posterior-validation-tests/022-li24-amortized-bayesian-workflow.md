# Amortized Bayesian Workflow

Li, Vehtari, Bürkner, Radev, Acerbi & Schmitt, 2024 (v3 2026), Transactions on Machine Learning Research · [arXiv:2409.04332](https://arxiv.org/abs/2409.04332)

**Evidence: B.** Peer-reviewed (TMLR; v3 states "Accepted in Transactions on Machine Learning Research");
7 citations in 2.1 yr (Semantic Scholar, 2026-10-22).

## What they do

The paper sets out a workflow for running a trained amortized posterior estimator on many observed datasets. It checks each dataset and escalates to slower, more accurate methods only where needed.

- **Training phase.** Check the estimator on simulated data with SBC on marginal projections, read with the graphical ECDF test of Säilynoja et al. (2022). Add parameter-recovery plots of estimates against true values. The authors say recovery plots help "mitigate a known failure mode of SBC with marginal projections as test quantities, in which the posterior approximation simply recovers the prior". Retrain if either check fails.
- **Step 1: amortized draws + OOD test.** Compute a summary statistic s(y), learned or hand-made. Take the Mahalanobis distance of s(y_obs) to the training summaries. Flag datasets whose distance lies in the right α tail of the training distances. Accept the amortized draws for the rest.
- **Step 2: PSIS.** For flagged datasets, importance-weight the amortized draws by p(y|θ)p(θ)/q(θ|y), with Pareto smoothing. Accept if Pareto-k̂ ≤ 0.7. This needs the likelihood and q's density.
- **Step 3: MCMC.** If k̂ > 0.7, run ChEES-HMC with many parallel chains initialised from the amortized or PSIS draws. Accept if nested R̂ < 1.01.

## What they find

Table 1, accepted datasets per step, and total time against running ChEES-HMC on every dataset:

| Problem | Step 1 | Step 2 | Step 3 | Total | Time (min) | Baseline (min) |
|---|---|---|---|---|---|---|
| GEV (3 params, 65 obs) | 523/1000 | 357/477 | 87/120 | 967/1000 | 15 | 85 |
| Bernoulli GLM (10 params) | 9519/10000 | 425/481 | 56/56 | 10000/10000 | 5 | 688 |
| Psychometric curve | 7213/8526 | 1215/1313 | 69/98 | 8497/8526 | 37 | 2217 |
| Decision model | 13498/15000 | 827/1502 | 554/675 | 14879/15000 | 659 | 11594 |

- **Speedup** over full ChEES-HMC: about 5×, 120×, 60× and 15×.
- **Accuracy.** Against well-tuned NUTS references, rejected amortized draws are markedly worse than accepted ones (W₁ and MMTV). PSIS-corrected draws are close to ChEES-HMC quality. Accepted Step-1 draws are less accurate than PSIS or HMC but are good for most datasets.
- **The GEV test sets** are deliberately simulated from a 2× wider prior, to create out-of-distribution data. The GLM test sets are all in-distribution.
- **Limits of the OOD test** (Appendix D, about 1000 datasets per task). The Mahalanobis distance correlates positively with posterior error when OOD data are present. For the in-distribution GLM it is uninformative, and the correlation is negative. Some low-distance datasets still get poor posteriors. Results are not very sensitive to α between 0.01 and 0.5; α = 0.05 is the default.

## Implementation details

- **Step 1 defaults.** α = 0.05, 10,000 training simulations for the distance reference, 2000 amortized draws per dataset. The cost is far below a second per dataset once trained.
- **PSIS threshold.** k̂ ≤ 0.7. With fewer than 2000 draws, the threshold is min(1 − 1/log₁₀ S, 0.7). PSIS needs a mass-covering proposal.
- **ChEES-HMC.** 2048 chains (16 superchains × 128 subchains), 200 warmup steps and a single sampling step, on a GPU.
- **Metrics.** W₁ on the joint and the mean marginal total variation (MMTV) in [0, 1]. MMTV below 0.2 is their rule of thumb for a good approximation.
- **Alternatives discussed.** Posterior SBC needs extra simulations per dataset and an estimator trained on doubled inputs. L-C2ST is "very sensitive" to classifier design, classifier calibration and how the simulation budget is split.
- **Code.** github.com/pipme/amortized-Bayesian-workflow.

## Caveats

- Only the training-phase checks and Step 1 are likelihood-free. Step 2 needs p(y|θ) and log q(θ|y). Step 3 needs the likelihood and its gradient.
- Step 1 tests whether y is out of distribution, not whether q is accurate. For in-distribution data, a wrong posterior passes Step 1.
- The training-phase SBC on marginal projections cannot detect a posterior equal to the prior. The paper relies on parameter-recovery plots for that, which are visual, not a test.
- Too-wide posteriors are not singled out as a failure mode.
- The problems have low-dimensional parameters (3 for GEV, 10 for the GLM) and tractable likelihoods.
