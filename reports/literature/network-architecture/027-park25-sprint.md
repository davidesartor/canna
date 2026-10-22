# SPRINT: Sparse-Dense Residual Fusion for Efficient Diffusion Transformers

Park et al., 2025, ICLR 2026 · [arXiv:2510.21986](https://arxiv.org/abs/2510.21986)

**Evidence: B.** Peer-reviewed; 6 citations in 1.0 yr (Semantic Scholar, 2026-10-19).

## What they do

2 dense blocks, then 25% of tokens through the middle 8 blocks, then fuse, then 2 dense blocks.

## What they find

- SiT-B: 75% drop gives 27.5 vs 54.1 at 0%. Without the dense path: 85.1.
- Split 2-8-2 gives 27.5; 5-2-5 gives 49.2.
- A 20K full-token fine-tune recovers 94%.
- Largest step in the Speedrun stack.

## Implementation details

Structured token selection plus a dense residual path.

## Caveats

87.5% drop gives 50.2.
