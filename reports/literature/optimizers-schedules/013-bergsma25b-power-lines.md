# Power Lines: Scaling Laws for Weight Decay and Batch Size in LLM Pre-training (Bergsma et al., 2025)

[arXiv:2505.13738](https://arxiv.org/abs/2505.13738), preprint. Related:
[Wang & Aitchison 2024](https://arxiv.org/abs/2405.13698).

**Evidence: A.** Peer-reviewed (Neural Information Processing Systems); 44 citations in 1.4 yr
(Semantic Scholar, 2026-10-19).

## What they do

- Treat AdamW's weights as an exponential moving average of updates, with a timescale of
  1/(ηλ) steps (η the lr, λ the weight decay). In tokens, that is τ = B/(ηλD).
- Fit how the optimal τ scales with batch size and tokens per parameter.

## What they find

- **The timescale τ is the quantity to tune**, not η and λ separately.
- Its optimum is a roughly fixed fraction of training, shifting predictably with tokens per
  parameter. Wang & Aitchison reach the same rule: keep 1/(ηλ) a fixed fraction of the run.

## Limits

- AdamW on LLMs. Muon with decay has the same 1/(ηλ) timescale, but it is not tested here.
