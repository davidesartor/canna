# Optimization Benchmark for Diffusion Models on Dynamical Systems (Schaipp et al., 2025)

[arXiv:2510.19376](https://arxiv.org/abs/2510.19376), preprint. The closest optimizer benchmark
to a scientific-inference diffusion model.

**Evidence: C.** Not peer-reviewed; 7 citations in 1.0 yr (Semantic Scholar, 2026-10-19). Its
claims are the authors' alone: a hypothesis, not a result.

## What they do

- Train a 23M-parameter U-Net denoiser (DDPM) on Kolmogorov-flow (Navier–Stokes) trajectories.
- Compare tuned AdamW, Muon, SOAP and schedule-free AdamW, each with its lr and decay tuned.
  The scheduled baselines use cosine or WSD.
- Score both the final training loss and the generated trajectories.

## What they find

- **Muon and SOAP:** 18% lower final loss than AdamW. AdamW given 2× the steps still loses.
  - SOAP is best per step.
  - Best Muon setting: lr 1e-3, weight decay 0.032.
- **Schedule-free AdamW:**
  - better early, slightly worse at the end;
  - its samples are visibly worse at similar loss;
  - the authors blame the missing lr annealing, and adding a cooldown repairs part of the gap.
- **Loss and sample quality can disagree.** WSD's loss-optimal lr gives worse samples than
  cosine.

## Limits

- One dataset and one model size.
- SF is the original SF-AdamW, not SF-Muon or ScheduleFree+.
