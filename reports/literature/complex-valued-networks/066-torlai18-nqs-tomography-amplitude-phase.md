# Neural-network quantum state tomography

Torlai, Mazzola, Carrasquilla, Troyer, Melko, Carleo, 2018, Nature Physics 14, 447 ·
[arXiv:1703.05334](https://arxiv.org/abs/1703.05334)

**Evidence: A.** Peer-reviewed (Nature Physics); 929 citations in 9.6 yr (Semantic Scholar,
2026-10-19).

## What they do

The paper reconstructs a quantum state from measurement samples (tomography) with RBMs.
The complex wavefunction is split explicitly:

ψ(σ) = √(p_λ(σ)/Z_λ) · exp(i φ_μ(σ)/2), with φ_μ = log p_μ(σ).

There are two RBM parameter sets on a shared visible layer: λ for the amplitude, μ for the
phase. **Training runs in two stages:**

1. Fit λ by standard RBM likelihood training on samples in the reference σᶻ basis, which
   only constrains |ψ|².
2. Freeze λ and fit μ on samples from **rotated measurement bases**. Only there do relative
   phases show up in the measured distribution. For example, rotating neighbours to X·X
   gives |Ψ|² ∝ cos(θ_{j+1} − θ_j), and X·Y gives ∝ sin(θ_{j+1} − θ_j).

## What they find

- **Phase-augmented W state, N = 20:** overlap 0.997 with 2N − 1 bases and 6400 samples per
  basis.
- **Efficiency:** for N = 8, brute-force tomography needs about 10⁶ measurements, while the
  RBM reaches comparable accuracy with about 100.
- **Ground states** of the 2D TFIM and XXZ (N = 144) and the 1D critical TFIM (N = 100) are
  real and positive, so they need only the σᶻ basis. Correlations are reproduced "with high
  accuracy", and second Rényi entropies at N = 20 agree with exact diagonalisation.
- **Long-range Ising quench:** phases are reconstructed at N = 12 with 2N + 1 bases.
- **Phase training is harder.** It needed natural-gradient descent ("more effective, though
  at the cost of increased computational resources"), with a diagonal shift to stabilise
  the noisy Fisher matrix.

## Implementation details

- Amplitude and phase have separate parameters and separate losses. The phase loss is the
  likelihood of rotated-basis outcomes given the frozen amplitudes.
- The number of bases needed scales polynomially with system size for the states of
  interest.

## Caveats

- The phase is learnable here only because the data contain phase-sensitive measurements
  (rotated bases). From σᶻ samples alone the phase is unidentifiable. This is a
  data-identifiability point, distinct from the optimisation-difficulty failures in files
  61–63.
- Small to moderate systems. The overlap is reported for the W state, not for every
  experiment.
- Details are from the ar5iv HTML.
