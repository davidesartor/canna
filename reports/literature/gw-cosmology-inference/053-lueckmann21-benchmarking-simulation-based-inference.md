# Benchmarking Simulation-Based Inference

Lueckmann, Boelts, Greenberg, Gonçalves, Macke, 2021, AISTATS 2021 · [arXiv:2101.04653](https://arxiv.org/abs/2101.04653)

**Evidence: A.** Peer-reviewed (International Conference on Artificial Intelligence and
Statistics); 302 citations in 5.8 yr (Semantic Scholar, 2026-10-19).

## What they do

sbibm. Ten tasks with reference posteriors (Gaussian Linear and its uniform-prior variant, 10-D; SLCP, 5 parameters with 4 modes; SLCP-Distractors with 92 nuisance outputs; Bernoulli GLM and GLM-Raw; Gaussian Mixture; Two Moons; SIR; Lotka-Volterra). Methods: REJ-ABC, SMC-ABC, NLE/SNLE, NPE/SNPE (APT atomic), NRE/SNRE, at 10³, 10⁴ and 10⁵ simulations, with 10 observations per task.

## What they find

- **Metric choice is critical.** C2ST (5-fold cross-validated MLP classifier, 2 hidden layers of 10× the data dimension; 0.5 is perfect, 1 is worst) was chosen over MMD, NLTP, median-distance PPCs and KSD.
  - On Two Moons, REJ-ABC had a lower MMD than SNLE (0.00729 vs 0.00772), yet a classifier separates REJ-ABC from the reference at 0.794 accuracy versus 0.502 for SNLE.
  - The median-heuristic MMD kernel fails on multimodal posteriors.
  - NLTP correlates with C2ST at only 0.3 (Two Moons) and 0.6 (SLCP).
- **No uniformly best algorithm.** Sequential methods help on most tasks, with diminishing returns as the budget grows. Neural methods beat ABC at small and moderate budgets. SLCP and Lotka-Volterra are not solved by any method within budget.
- **NSF vs MAF:** neural spline flows beat MAFs for (S)NPE, especially on Gaussian Mixture, Two Moons and SIR, where NSF removed the density-transformation artifacts. For (S)NLE, NSF was worse than MAF.

## Implementation details

NPE used NSF with 5 transforms, 2 residual blocks of 50 units, ReLU, and 10 bins. NLE used MAF with 5 transforms of 2×50 tanh and batch norm, sampled by MCMC with 100 chains and 250 warm-up steps. Everything ran on single CPU cores; GPUs gave no benefit for these small networks. SNPE had the longest runtime.
- Box 1 recommendations: use amortized NPE for repeated inference on many observations; use sequential methods for expensive simulators; use the likelihood or simulator internals when available.

## Caveats

Only low-dimensional θ and x, no images. Tuning reflects the authors' expertise. Budget is counted in simulations, not wall clock. Only tasks with reference posteriors are included.
