# The Surprising Agreement Between Convex Optimization Theory and Learning-Rate Scheduling for Large Model Training (Schaipp et al., 2025)

[arXiv:2501.18965](https://arxiv.org/abs/2501.18965), ICML 2025.

**Evidence: B.** Peer-reviewed (International Conference on Machine Learning); 39 citations in
1.7 yr (Semantic Scholar, 2026-10-19).

## What they do

- Compare the nonsmooth-convex suboptimality bound for SGD under different lr schedules with
  real LLM training curves.

## What they find

- The bound reproduces the cooldown benefit, with no log term.
- It predicts how to extend a run (continue at a lower lr, not restart at the peak) and how
  to transfer the lr between schedules.
