# Exploring the Deep Fusion of Large Language Models and Diffusion Transformers for Text-to-Image Synthesis

Tang et al., 2025 · [arXiv:2505.10046](https://arxiv.org/abs/2505.10046)

**Evidence: B.** Peer-reviewed (Computer Vision and Pattern Recognition); 18 citations in 1.4
yr (Semantic Scholar, 2026-10-19).

## What they do

Text tokens are causal and never attend to image tokens, so their K/V are cached across all steps.

## What they find

CC12M 512², 300K steps:
| Design | GenEval | DPG | FID | Latency |
|---|---|---|---|---|
| Self-attention | 0.42 | 73.9 | 26.16 | 1.75 s |
| Cross-attention | 0.49 | 76.3 | 24.00 | 1.86 s |
| Deep fusion | 0.51 | 76.6 | 27.33 | 1.66 s |
Dropping timestep conditioning altogether gives about 20% fewer parameters, GenEval 0.49, and the best FID, 21.27.

## Implementation details

One-way attention from condition to generation. No t on the condition side.

## Caveats

Deep fusion has the worst FID of the three despite the best alignment.
