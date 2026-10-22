# Muown: Row-Norm Control for Muon Optimization (Lion et al., 2026)

[arXiv:2605.10797](https://arxiv.org/abs/2605.10797), preprint.

**Evidence: C.** Not peer-reviewed; 6 citations in 0.4 yr (Semantic Scholar, 2026-10-19). Its
claims are the authors' alone: a hypothesis, not a result.

## What they do

- Split each weight matrix into per-row magnitudes and a direction. Muon updates the direction;
  the row magnitudes are learned separately.

## What they find

- Less sensitivity to weight decay.
- Row scale is decoupled from direction learning, so rows that need a different size are not
  pulled to a common norm.

## Limits

- LLMs only.
