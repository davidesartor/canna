# Harmonic Networks: Deep Translation and Rotation Equivariance

Worrall, Garbin, Turmukhambetov & Brostow, 2017, CVPR 2017 ·
[arXiv:1612.04642](https://arxiv.org/abs/1612.04642)

**Evidence: A.** Peer-reviewed (CVPR 2017); 803 citations in 9.8 yr (Semantic Scholar,
2026-10-19).

## What they do

H-Nets: CNNs equivariant to continuous 2D rotations using complex-valued features. Filters are
circular harmonics `W_m(r, φ) = R(r) e^{i(mφ + β)}` (learned radial profile R, learned phase
offset β, rotation order m). Rotating the input by θ multiplies a response of order m by
`e^{imθ}`, so rotation becomes a phase shift of complex feature maps:

- streams of fixed total rotation order (m ∈ {0, 1} in practice), where orders add along a
  path through the network;
- **C-ReLU**: `ReLU(|X| + b) · e^{i arg X}` — pointwise nonlinearities act on magnitude only,
  preserving phase and hence equivariance (same pattern as modReLU, Arjovsky et al. 2016);
- magnitude-based pooling and batch norm; final layer takes magnitudes (rotation-invariant).

## What they find

- Rotated MNIST: test error 1.69% (33k params) vs P4CNN 2.28% (25k), plain CNN 5.03% (22k),
  CNN with rotation augmentation 3.50%.
- BSD500 boundary detection: ODS 0.726 with 116k params vs HED 0.640 at 2346k (best among
  non-ImageNet-pretrained models in their table).

## Implementation details

- A complex cross-correlation costs four real ones; equating multiply counts gives the rule
  of thumb `i_CNN = 2f · i_H` channels for equal compute (f = filter size).

## Caveats

- The symmetry is spatial rotation of the domain, mapped to a phase of the features; the
  phase-of-features idea and magnitude-only nonlinearities carry over, but the input itself is
  real.
- Small benchmarks; equivariance is approximate after discretisation of filters.
