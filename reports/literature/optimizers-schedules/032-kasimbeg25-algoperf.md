# Accelerating Neural Network Training: An Analysis of the AlgoPerf Competition (Kasimbeg et al., 2025)

[arXiv:2502.15015](https://arxiv.org/abs/2502.15015), ICLR 2025. The benchmark itself is
Dahl et al., [arXiv:2306.07179](https://arxiv.org/abs/2306.07179).

**Evidence: B.** Peer-reviewed (International Conference on Learning Representations); 38
citations in 1.7 yr (Semantic Scholar, 2026-10-19).

## What they do

- Analyse the AlgoPerf competition: fixed workloads, wall-clock to target, with an external-
  tuning track and a self-tuning track.

## What they find

- Distributed Shampoo won external tuning.
- Schedule-Free AdamW won self-tuning: its strength is robustness without tuning, not peak
  quality.
