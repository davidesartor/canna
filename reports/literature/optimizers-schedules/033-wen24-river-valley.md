# Understanding Warmup-Stable-Decay Learning Rates: A River Valley Loss Landscape Perspective (Wen et al., 2024)

[arXiv:2410.05192](https://arxiv.org/abs/2410.05192), ICLR 2025.

**Evidence: A.** Peer-reviewed; 86 citations in 2.0 yr (Semantic Scholar, 2026-10-19).

## What they do

- Model the loss as a river valley: a flat direction along the river, steep walls across it.

## What they find

- A high lr makes progress along the river while oscillating across it.
- The decay removes the cross-river oscillation, which is when the hidden progress shows up in
  the loss.
