# Rotational Equilibrium: How Weight Decay Balances Learning Across Neural Networks (Kosson et al., 2023)

[arXiv:2305.17212](https://arxiv.org/abs/2305.17212), ICML 2024.

**Evidence: B.** Peer-reviewed (International Conference on Machine Learning); 60 citations in
3.4 yr (Semantic Scholar, 2026-10-19).

## What they do

- Analyse scale-invariant weights (weights followed by a normalisation) under weight decay.

## What they find

- The weight norm reaches an equilibrium where the angular update per step is about
  √(2ηλ).
- The balance of angular updates across layers is what decay buys; it acts as an effective
  learning-rate control, not a regulariser.
- Weights that are not scale-invariant have no such equilibrium: decay just shrinks them.
