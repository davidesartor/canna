# Analog Bits: Generating Discrete Data using Diffusion Models with Self-Conditioning

Chen et al., 2022, ICLR 2023 · [arXiv:2208.04202](https://arxiv.org/abs/2208.04202)

**Evidence: A.** Peer-reviewed (International Conference on Learning Representations); 544
citations in 4.2 yr (Semantic Scholar, 2026-10-19).

## What they do

Self-conditioning: feed the previous x̂₀ estimate back as input.

## What they find

Continuous ImageNet-64, 20 steps: ε-pred 6.10 → 4.24, x₀-pred 12.13 → 6.94. Under 25% extra training cost.

## Implementation details

50% of training steps use self-conditioning, with stop-gradient.

## Caveats

—
