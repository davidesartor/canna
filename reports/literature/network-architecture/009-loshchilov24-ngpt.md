# nGPT: Normalized Transformer with Representation Learning on the Hypersphere

Loshchilov et al., 2024, ICLR 2025 · [arXiv:2410.01131](https://arxiv.org/abs/2410.01131)

**Evidence: A.** Peer-reviewed (International Conference on Learning Representations); 80
citations in 2.0 yr (Semantic Scholar, 2026-10-19).

## What they do

All hidden vectors and weights on the unit hypersphere, with a learnable QK scale s_qk.

## What they find

4×, 10× and 20× fewer steps at 1k, 4k and 8k context, but 60–80% more cost per step.

## Implementation details

A learned per-head QK scale.

## Caveats

No diffusion replication.
