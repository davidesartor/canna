# The AdEMAMix Optimizer: Better, Faster, Older (Pagliardini et al., 2024)

[arXiv:2409.03137](https://arxiv.org/abs/2409.03137), ICLR 2025.

**Evidence: B.** Peer-reviewed (International Conference on Learning Representations); 58
citations in 2.1 yr (Semantic Scholar, 2026-10-19).

## What they do

- Mix Adam's fast gradient EMA with a very slow one (β₃ ≈ 0.9999).

## What they find

- A 1.3B model trained on 101B tokens matches AdamW trained on 197B.
- Best optimizer overall in the LLM benchmark of [030](030-semenov25-llm-optimizer-benchmark.md).

## Limits

- Available as `optax.contrib.ademamix` and `simplified_ademamix`.
