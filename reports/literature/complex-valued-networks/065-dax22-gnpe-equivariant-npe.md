# Group equivariant neural posterior estimation

Dax, Green, Gair, Deistler, Schölkopf, Macke, 2022, ICLR ·
[arXiv:2111.13139](https://arxiv.org/abs/2111.13139)

**Evidence: B.** Peer-reviewed (ICLR 2022); 40 citations in 4.9 yr (Semantic Scholar,
2026-10-19).

## What they do

GNPE builds a known joint symmetry of (parameters, data) into NPE **without changing the
architecture**. A group G acts on θ and x, and the "pose" g_θ is the part of θ that G moves.

- **Standardise the pose.** Transform the data by the inverse of an estimated pose,
  x′ = T_{ĝ⁻¹} x, so the network always sees data in a canonical frame.
- **Blur the pose estimate.** The proxy is ĝ = g_θ ε with ε ~ κ, a narrow symmetric kernel.
  "Without blurring, GNPE would not work."
- **Sample by Gibbs-like iteration.** Alternate θ ~ q(θ | x′, ĝ) and ĝ ~ κ(g_θ⁻¹ ĝ), starting
  from an ordinary NPE estimate of the pose.
- **Exact vs approximate symmetries.** For an exact equivariance the network is q(θ′ | x′)
  and the posterior is equivariant by construction. For an approximate one the network is
  also conditioned on ĝ.

For gravitational waves the group is the detector time shifts. In the frequency domain "time
translations act by multiplication": a shift Δt multiplies bin f by e^{−2πifΔt}. Pose
standardisation is therefore a **frequency-dependent phase rotation of the complex data**.

- The global arrival time is an exact symmetry.
- The relative detector delays, set by sky position, are only approximate.

## What they find

- **Toy damped oscillator** with the excitation time as pose: GNPE matches an NPE with a
  convolutional embedding and "far outperforms" standard NPE (c2st).
- **8 GWTC-1 events:** standard NPE reaches c2st scores up to 0.71. GNPE in accurate mode
  (±1 ms kernel, 30 iterations, about 60 s) gets below 0.55 on all eight. Fast mode (±3 ms,
  one iteration, about 2 s) is coarser.
- **Chained NPE** (pose first, then the rest) fails on GW170814 because its first-stage pose
  is wrong. Iteration avoids this.
- **Speed:** about three orders of magnitude faster than MCMC.

## Implementation details

- Convergence is monitored by the JS divergence between pose posteriors of successive
  iterations, with burn-in and thinning across parallel chains.
- Pose standardisation is a pre-processing operation on the complex frequency series, so
  any real-valued network can sit behind it.
- Training takes 16–18 days on a V100 for the GW network.

## Caveats

- The method needs the transformation T_g to act on the data in closed form. Time shifts
  qualify.
- The paper applies GNPE only to time shifts. The reference phase φc appears only as a
  prior in their Table D.1. Our note, not the paper's: for a single-mode signal φc is also a
  global phase rotation of the data, but with higher modes or precession it is not one
  rotation. DINGO-IS later handled φc separately, with a likelihood grid (file 60).
- Inference cost grows with iterations: O(10) typical, 30 in accurate mode.
- Gibbs sampling is approximate, and the network must stay accurate over the blurred pose
  range.
