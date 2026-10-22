# LayerSync: Self-aligning Intermediate Layers

Haghighi et al., 2025, ICLR 2026 · [arXiv:2510.12581](https://arxiv.org/abs/2510.12581)

**Evidence: B.** Peer-reviewed; 9 citations in 1.0 yr (Semantic Scholar, 2026-10-19).

## What they do

A cosine loss to the model's own deeper layer, with stop-gradient.

## What they find

- SiT-B at 80 epochs: 36.19 → 30.00.
- Audio FAD 0.251 → 0.199. Motion (8-layer MDM) 0.52 → 0.48.
- Even random layer pairs help.

## Implementation details

Robust to λ. Parameter-free.

## Caveats

Generic convergence gains only.
