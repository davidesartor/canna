# The universal approximation theorem for complex-valued neural networks

Voigtlaender, 2023, Applied and Computational Harmonic Analysis ·
[arXiv:2012.03351](https://arxiv.org/abs/2012.03351)

**Evidence: B.** Peer-reviewed (ACHA); 79 citations in ~6 yr (Semantic Scholar,
2026-10-19).

## What they do

Characterises exactly which activations make complex feedforward networks
(`σ(b + wᵀz)`, complex weights, σ applied componentwise) universal approximators of
continuous functions `ℂ^d → ℂ` on compacts, for locally bounded σ whose discontinuity set has
null closure.

## What they find

- **Shallow (one hidden layer)**: universal **iff σ is not almost-everywhere polyharmonic**
  (no m with `Δ^m σ = 0`). So holomorphic activations are *never* universal in shallow nets
  (they are harmonic); any bounded, non-constant σ is; shallow nets with sin, sinh, tan, tanh
  are **not** universal, contrary to earlier claims.
- **Deep (≥ 2 hidden layers)**: universal unless σ is a.e. a polynomial in `(z, z̄)`, or σ or
  σ̄ is a.e. an entire holomorphic function (necessary and sufficient for continuous σ).
  Deep complex nets admit strictly more universal activations than shallow ones — unlike the
  real case, where "not a polynomial" is the condition at every depth.
- Corollary for the usual activations (by the criteria; the paper does not list them):
  modReLU (b ≠ 0), CReLU, zReLU and cardioid are not a.e. equal to any smooth function (kink,
  or non-smooth at 0), and bounded amplitude–phase squashers are bounded and non-constant,
  so all are universal even in shallow nets (our reading of the theorem).
  Pure holomorphic choices (`tanh z`, `z²`) are the ones to avoid.

## Implementation details

Pure theory, no experiments, no rates.

## Caveats

- Qualitative density results only: say nothing about efficiency, parameter count or
  trainability, hence nothing about complex vs real at matched compute.
- A complex network is in particular a real network on `ℝ^{2d}` with constrained weights, so
  universality alone gives a complex net no expressive edge over a real one.
