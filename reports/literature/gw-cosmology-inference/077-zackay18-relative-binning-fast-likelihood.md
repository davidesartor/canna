# Relative Binning and Fast Likelihood Evaluation for Gravitational Wave Parameter Estimation

Zackay et al., 2018, arXiv letter [astro-ph.IM] · [arXiv:1806.08792](https://arxiv.org/abs/1806.08792)

**Evidence: A.** Not peer-reviewed; 104 citations in 8.3 yr (Semantic Scholar, 2026-10-19).

## What they do

- Near the posterior, the ratio r(f) = h(f)/h₀(f) to a fiducial (near-max-likelihood) waveform is smooth. In each frequency bin, r ≈ r₀ + r₁(f − f_m).
- Precompute full-resolution summary data per bin: A₀, A₁ = Σ d h₀* /(S/T) × (1, f − f_m) and B₀, B₁ = Σ |h₀|²/(S/T) × (1, f − f_m).
- Then Z[d, h] ≈ Σ_b (A₀ r₀* + A₁ r₁*) and Z[h, h] ≈ Σ_b (B₀|r₀|² + 2 B₁ Re[r₀ r₁*]).
- The complex Z[d, h] lets the constant phase φ_c be marginalised or maximised analytically.

## What they find

- Waveform evaluations per likelihood (Table 1): 63 for relative binning, 1740 for ROQ, ~10⁴ for multibanding, ~10⁷ for the full FFT grid. That is ~10⁴× faster than naive and ~10× faster than ROQ.
- Error model: Δ ln L_bin ≈ β |ln L_max − ln L|, i.e. it grows away from the peak. β < 0.01 with 62 bins (χ = 1, ε = 0.5 rad). Doubling to 104 bins improves accuracy quadratically.
- The fiducial was 0.07 below the maximum of ln L_max ≈ 573. About 3×10⁴ posterior points were checked.
- SVD of the binned ratios: 10–20 components would suffice.

## Implementation details

- Bin edges chosen with no physics knowledge: bound the worst-case PN phase drift δΨ_max(f) = 2πχ Σ (f/f*_i)^{γ_i} sgn γ_i, with γ ∈ {−5/3, −1, −2/3, 1, 5/3}, so that each bin changes by less than ε rad.
- If the phase error per bin is below 1 rad, raise the polynomial order instead of adding bins.
- Noise can be simulated directly in summary space. δA₀, δA₁ are complex Gaussian with covariances ⟨|δA₀|²⟩ = B₀, ⟨δA₀ δA₁*⟩ = B₁, ⟨|δA₁|²⟩ = B₂. This makes fast injection studies possible without per-bin FFT noise.
- Higher modes need separate bins per mode.

## Caveats

needs a good fiducial (maximum likelihood), so it is not amortised and the error grows away from the fiducial. For an NPE it is a compression relative to a reference; that reference must be known per observation (cf. heterodyning in DINGO-style pipelines).
