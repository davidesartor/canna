# Solving the quantum many-body problem with artificial neural networks

Carleo, Troyer, 2017, Science 355, 602 · [arXiv:1606.02318](https://arxiv.org/abs/1606.02318)

**Evidence: A.** Peer-reviewed (Science); 2450 citations in 10.3 yr (Semantic Scholar,
2026-10-19).

## What they do

This paper introduces neural quantum states (NQS). The many-body wavefunction is a
restricted Boltzmann machine with **complex weights**:

Ψ(σ) = exp(Σ_i a_i σ_i) · Π_j 2 cosh(b_j + Σ_i W_ji σ_i), with a, b, W ∈ ℂ.

Complex parameters let one network carry "both the amplitude and the wave-function's
phase". The ansatz is holomorphic in its parameters, so log Ψ has a single complex
derivative per parameter.

- **Ground states:** energy minimised by stochastic reconfiguration (SR, a natural-gradient
  method) on Metropolis samples of |Ψ|².
- **Dynamics:** time-dependent VMC.
- **Systems:** TFI and antiferromagnetic Heisenberg (AFH) chains of 80 sites, and the AFH on
  a 10 × 10 square lattice.
- **Hidden density** α = M/N, with translation symmetry imposed on the weights.

## What they find

- **1D TFI at criticality:** about one part per million in energy at modest α. The error
  falls roughly as a power law in α.
- **1D AFH:** comparable accuracy at α = 4, orders of magnitude better than spin-Jastrow.
  DMRG needs bond dimension about 160 to match.
- **2D AFH, 10 × 10:** on par with EPS/PEPS at α ≈ 4, and the best variational energies
  then known for larger α.
- **Quenches** (TFI and AFH) match exact or t-DMRG dynamics at α = 4. The complex
  representation "effectively solves the phase-problem" of stochastic dynamics.

## Implementation details

- The regularisation shift in SR decays as max(100 · 0.9ᵖ, 1e-4).
- Weights are initialised to small random complex numbers.
- Each SR step costs O(αN²).

## Caveats

- **Easy sign structure only.** The benchmarks are unfrustrated, with a ground-state sign
  structure known in closed form (Marshall). The text extracted for this note does not
  discuss how the sign is handled.
- **Frustration is harder.** Later work found holomorphic complex-weight NQS struggle once
  the sign structure is nontrivial, and recommends separate real amplitude and phase
  networks (files 61–63).
- Exact per-α errors for 2D are only in Fig. 3.
- Details are from the ar5iv HTML.
