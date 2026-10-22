# CMuon: Accelerating and Stabilizing Diffusion Transformer Training via Chunked Momentum Orthogonalization

Chen et al., 2026, ECCV 2026 · [arXiv:2608.02502](https://arxiv.org/abs/2608.02502)

**Evidence: B.** Peer-reviewed; 3 citations in 0.2 yr (Semantic Scholar, 2026-10-19).

## What they do

Orthogonalize fused QKV, gate-up and adaLN matrices chunk by chunk.

## What they find

- DiT-B FID at epoch 200: AdamW 3.49, Muon 3.31, CMuon 3.03.
- At 512², CMuon 2.38 is *worse* than Muon 2.16.
- Learning-rate rule (vanilla vs Moonlight) swings FID from 8.73 to 3.31.

## Implementation details

Treat fused matrices as separate blocks for Newton–Schulz.

## Caveats

Mixed result at 512.
