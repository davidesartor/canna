# High-Performance Large-Scale Image Recognition Without Normalization, NFNets / AGC (Brock et al., 2021)

[arXiv:2102.06171](https://arxiv.org/abs/2102.06171), ICML 2021.

**Evidence: A.** Peer-reviewed (International Conference on Machine Learning); 657 citations in
5.7 yr (Semantic Scholar, 2026-10-19).

## What they do

- Adaptive gradient clipping (AGC): clip each unit's gradient when ‖g‖/‖w‖ exceeds a
  threshold.

## What they find

- AGC lets normaliser-free networks train stably at large batch and high lr.
- The clip is per-unit and scale-aware, unlike a global norm clip.

## Limits

- Available as `optax.adaptive_grad_clip`.
