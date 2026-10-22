# Connections between Schedule-Free Optimizers, AdEMAMix, and Accelerated SGD Variants (Morwani et al., 2025)

[arXiv:2502.02431](https://arxiv.org/abs/2502.02431), preprint.

**Evidence: C.** Not peer-reviewed; 10 citations in 1.7 yr (Semantic Scholar, 2026-10-19). Its
claims are the authors' alone: a hypothesis, not a result.

## What they do

- Put schedule-free and AdEMAMix in one family of accelerated-SGD variants.

## What they find

- Both decouple the momentum coefficient from the weight on the current gradient.
- Simplified-AdEMAMix uses a single momentum buffer and matches AdEMAMix.
- So schedule-free's gains are reachable without its x/y/z bookkeeping, with an ordinary
  schedule on top.
