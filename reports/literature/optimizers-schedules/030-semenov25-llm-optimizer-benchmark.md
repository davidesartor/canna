# Benchmarking Optimizers for Large Language Model Pretraining (Semenov et al., 2025)

[arXiv:2509.01440](https://arxiv.org/abs/2509.01440), preprint.

**Evidence: B.** Not peer-reviewed; 61 citations in 1.1 yr (Semantic Scholar, 2026-10-19).

## What they do

- Benchmark a wide set of optimizers on LLM pretraining across batch sizes and run lengths.

## What they find

- AdEMAMix is best overall. D-Muon (Muon with decay on the 2-D weights) is robust.
- SF-AdamW is weak at small batch and needs a long (~25%) warmup.
- Decaying the lr to 0.01× (0.001× for linear) beats 0.1×.
- **β₂ = 0.999 beats 0.95 on long runs.**
- Weight decay ~0.1 is robust; 0 hurts.
- MARS is strong at large batch and weak at small; Lion and Sophia need large batch, and Sophia
  diverges on long horizons.

## Limits

- LLMs only.
