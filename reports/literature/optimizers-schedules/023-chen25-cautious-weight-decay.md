# Cautious Weight Decay (Chen et al., 2025)

[arXiv:2510.12402](https://arxiv.org/abs/2510.12402), preprint.

**Evidence: C.** Not peer-reviewed; 15 citations in 1.0 yr (Semantic Scholar, 2026-10-19). Its
claims are the authors' alone: a hypothesis, not a result.

## What they do

- Apply weight decay only on coordinates where the update and the weight have the same sign.

## What they find

- The method optimises the original loss rather than a regularised one, so decay no longer
  biases weights toward zero.
- Gains for AdamW, Lion and Muon.
