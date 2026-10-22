# Self-Consistent Flow: Unifying Velocity and Endpoint Prediction for Rectified Flow Models

Han et al., 2026, TMLR 2026 · [arXiv:2607.12171](https://arxiv.org/abs/2607.12171)

**Evidence: B.** Peer-reviewed (Trans. Mach. Learn. Res.); 3 citations in 0.3 yr (Semantic
Scholar, 2026-10-19).

## What they do

One network outputs v or x₁ via a mode bit. The two are tied by a consistency loss x̂₁ ≈ x_t + (1−t)·v̂.

## What they find

- Consistency weight w_c from 0 to 0.1: FID 11.14 → 9.85.
- Shared network 9.92 vs two networks 10.92.
- XL at 800K: 6.22 → 4.19.
- Switching from X-mode to V-mode at τ is flat over 0.3–0.7.

## Implementation details

X-mode early and V-mode late when sampling.

## Caveats

The paper is internally inconsistent on whether the x-loss weight is 1/(1−t) or 1/(1−t)². Not replicated.
