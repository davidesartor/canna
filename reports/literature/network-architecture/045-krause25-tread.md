# TREAD: Token Routing for Efficient Architecture-agnostic Diffusion Training

Krause et al., 2025, ICCV 2025 · [arXiv:2501.04765](https://arxiv.org/abs/2501.04765)

**Evidence: B.** Peer-reviewed (IEEE International Conference on Computer Vision); 23 citations
in 1.8 yr (Semantic Scholar, 2026-10-19).

## What they do

50% of tokens, chosen at random, skip layers 2 to L−4 and are re-injected. No new parameters; loss on all tokens.

## What they find

- DiT-B at 400K: 43.47 → 21.34. SiT-XL: 17.20 → 4.89.
- Dropout without re-injection: 32.00 vs TREAD 15.98.

## Implementation details

Route, do not drop.

## Caveats

Cliff at 75% routing (49.29). Needs a 75K unrouted fine-tune for CFG. Did not reproduce under SPRINT's protocol.
