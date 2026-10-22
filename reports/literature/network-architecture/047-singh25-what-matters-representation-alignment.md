# What matters for Representation Alignment: Global Information or Spatial Structure?

Singh et al., 2025, ICLR 2026 · [arXiv:2512.10794](https://arxiv.org/abs/2512.10794)

**Evidence: A.** Peer-reviewed; 99 citations in 0.9 yr (Semantic Scholar, 2026-10-19).

## What they do

iREPA. The target's spatial structure, not its global information, predicts the gain.

## What they find

Spatial self-similarity correlates with FID at |r| ≈ 0.85–0.89; linear-probe accuracy only at 0.26. A conv projector plus spatial norm helps, but the gain shrinks with training (400K: 7.76 → 7.52).

## Implementation details

Conv projector.

## Caveats

The gains fade.
