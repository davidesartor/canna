# Muon is Scalable for LLM Training, Moonlight (Liu et al., 2025)

[arXiv:2502.16982](https://arxiv.org/abs/2502.16982), preprint.

**Evidence: A.** Not peer-reviewed; 441 citations in 1.6 yr (Semantic Scholar, 2026-10-19).

## What they do

Two changes to make Muon scale to large LLMs:

1. **Decoupled weight decay** on the Muon matrices. Without it, weights and the RMS of layer
   outputs grow over long runs.
2. **Update RMS matched to AdamW's.** The orthogonalised update is scaled by 0.2·√max(A, B)
   for an A×B matrix, so every matrix gets an update of RMS ≈ 0.2·lr. AdamW's lr and decay
   then carry over unchanged. This is optax's `consistent_rms=0.2`.

## What they find

- About 2× the compute efficiency of AdamW, on 3B and 16B MoE models (Moonlight).
- Without decay, Muon's early lead over AdamW fades on long runs. With decay it holds.

## Limits

- LLMs only, without schedule-free.
