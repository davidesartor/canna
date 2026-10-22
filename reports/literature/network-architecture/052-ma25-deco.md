# DeCo: Frequency-Decoupled Pixel Diffusion for End-to-End Image Generation

Ma et al., 2025 · [arXiv:2511.19365](https://arxiv.org/abs/2511.19365)

**Evidence: B.** Not peer-reviewed; 65 citations in 0.9 yr (Semantic Scholar, 2026-10-19).

## What they do

A small pixel decoder (width 32, depth 3, patch 1) with a frequency-weighted FM loss.

## What they find

adaLN injection of the backbone features beats additive injection (FID 31.35 vs 36.02).

## Implementation details

Inject features per token through adaLN.

## Caveats

—
