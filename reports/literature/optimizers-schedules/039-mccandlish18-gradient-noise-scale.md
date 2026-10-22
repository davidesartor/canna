# An Empirical Model of Large-Batch Training (McCandlish et al., 2018)

[arXiv:1812.06162](https://arxiv.org/abs/1812.06162), preprint.

**Evidence: A.** Not peer-reviewed; 457 citations in 7.8 yr (Semantic Scholar, 2026-10-19).

## What they do

- Define the gradient noise scale B_noise = tr(Σ)/|G|².

## What they find

- B_noise predicts the critical batch size, above which a larger batch stops paying off.
- It grows as the loss falls, so the useful batch size grows during training.
