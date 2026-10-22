# Straight to Zero: Why Linearly Decaying the Learning Rate to Zero Works Best for LLMs (Bergsma et al., 2025)

[arXiv:2502.15938](https://arxiv.org/abs/2502.15938), preprint.

**Evidence: B.** Peer-reviewed (International Conference on Learning Representations); 38
citations in 1.7 yr (Semantic Scholar, 2026-10-19).

## What they do

- Compare linear decay to zero against the usual decay to 10% of peak, for LLMs at
  several sizes and tokens per parameter.
- Interpret AdamW's weights as a weighted average of past updates.

## What they find

- **Linear decay to zero beats decay to 10%.** The gap grows with tokens per parameter.
- **Why, in the averaging view:** a lower final lr means the final weights average over more
  updates, so less gradient noise remains.

## Limits

- LLMs with AdamW.
