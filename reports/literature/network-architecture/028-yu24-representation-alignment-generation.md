# Representation Alignment for Generation: Training Diffusion Transformers Is Easier Than You Think

Yu et al., 2024, ICLR 2025 · [arXiv:2410.06940](https://arxiv.org/abs/2410.06940)

**Evidence: A.** Peer-reviewed (International Conference on Learning Representations); 790
citations in 2.0 yr (Semantic Scholar, 2026-10-19).

## What they do

REPA. Align an intermediate DiT layer with a pretrained encoder.

## What they find

- Alignment depth: SiT-L aligned at depth 8 gives 10.0, at depth 16 gives 12.1, vanilla 18.8.
- Encoder: MAE 12.5, DINOv2 9.7.
- Model size: SiT-B 33.0 → 24.4; SiT-XL 17.2 → 7.9.

## Implementation details

Align about a third of the way in, through an MLP projector.

## Caveats

Needs an encoder. LISA has none, but the physics (matched-filter phasors) can serve.
