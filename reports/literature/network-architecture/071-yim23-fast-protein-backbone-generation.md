# Fast protein backbone generation with SE(3) flow matching

Yim et al., 2023 · [arXiv:2310.05297](https://arxiv.org/abs/2310.05297)

**Evidence: A.** Not peer-reviewed; 163 citations in 3.0 yr (Semantic Scholar, 2026-10-19).

## What they do

FrameFlow.

## What they find

An exp(−10t) rotation schedule is used at inference only; training with it made the task "too easy". Loss weight 1/(1−t)² clipped at t = 0.9.

## Implementation details

Inference-only per-group schedules.

## Caveats

—
