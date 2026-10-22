# Large Scale Diffusion Distillation via Score-Regularized Continuous-Time Consistency

Zheng et al., 2025 · [arXiv:2510.08431](https://arxiv.org/abs/2510.08431)

**Evidence: B.** Not peer-reviewed; 84 citations in 1.0 yr (Semantic Scholar, 2026-10-19).

## What they do

rCM. Its instability was traced to the oscillating trig time embedding.

## What they find

Fixed by fp32 time-embedding layers (as in Wan) and fp64 time warping. No numerical ablation extracted.

## Implementation details

Keep time embedding and modulation in fp32 inside a bf16 network.

## Caveats

Distillation setting.
