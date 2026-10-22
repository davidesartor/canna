# Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations (Hägele et al., 2024)

[arXiv:2405.18392](https://arxiv.org/abs/2405.18392), NeurIPS 2024.

**Evidence: A.** Peer-reviewed (Neural Information Processing Systems); 176 citations in 2.4 yr
(Semantic Scholar, 2026-10-19).

## What they do

- Compare cosine against a constant lr followed by a short cooldown (WSD), on LLMs at several
  scales.
- Compare stochastic weight averaging (SWA) and schedule-free as cooldown-free alternatives.

## What they find

- **Constant lr plus a cooldown of about 20% matches cosine.**
  - The 1−√ cooldown shape is best.
- **Averaging does not fully replace the cooldown.** SWA improves the constant-lr trajectory
  but does not match a cooldown.
- Schedule-free is competitive, but sensitive to β₂ and to momentum.

## Limits

- LLMs with AdamW only.
