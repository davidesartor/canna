# Marginalizing the likelihood function for modeled gravitational wave searches

Dhurandhar et al., 2017, arXiv preprint · [arXiv:1707.08163](https://arxiv.org/abs/1707.08163)

**Evidence: C.** Not peer-reviewed; 7 citations in 9.2 yr (Semantic Scholar, 2026-10-19). Its
claims are the authors' alone: a hypothesis, not a result.

## What they do

- (φ0, ψ, ι) are coordinates on SO(3) ≅ S³, and the ℓ = 2 Wigner matrix elements are the natural basis.
- Complex amplitudes: ℬ1 = A e^{−2iφ0} ((1+χ)/2)² e^{−2iψ} and ℬ2 = A e^{−2iφ0} ((1−χ)/2)² e^{+2iψ}, with ℬ3 = ℬ2*, ℬ4 = ℬ1*. These are a linear (unitary-like) map S of the JKS 𝒜^μ.
- Inversion: √A = √|ℬ1| + √|ℬ2|, χ = (√|ℬ1| − √|ℬ2|)/(√|ℬ1| + √|ℬ2|), ψ = −¼ arg(ℬ1/ℬ2).
- The log-likelihood is ln Λ = 2Re(ℬ1* y1) + 2Re(ℬ2 y2) − ½ζ(|ℬ1|² + |ℬ2|²) − Re(ℬ1 ℬ2* κ). Here ζ = |a|² + |b|² is real and κ = (a + ib)² is complex; the metric N is block-diagonal, which is a benefit over the real CPF amplitudes.

## What they find

- In polar form, ln Λ = −½ζ(R1² + R2²) + 2R1ρ1 cos(θ1 − ϕ1) + 2R2ρ2 cos(θ2 − ϕ2) − kR1R2 cos(θ1 − θ2 + 4η).
  - **Each phase θi = −2φ0 ∓ 2ψ (i.e. φ0 ± 2ψ up to sign and factor) has a von-Mises conditional with concentration 2Riρi.** The cross term couples only the difference (∝ ψ).
- Prior measure with the uniform physical prior: ∝ 1/√(R1R2) (Eq. 69).
- For κ ≠ 0, a unitary rotation diagonalises N with eigenvalues ζ ± k. This is a sky-dependent shift ψ → ψ − η, i.e. the "dominant polarisation frame" of Harry & Fairhurst.
- High-SNR B ≈ … × |z1 − z2|^{−3/2} |z1 + z2|^{−3/2} × exp(…), Eq. 88.
- Singularities:
  - (i) ζ = k, a property of the network/sky. For 2 detectors this happens on curves on the sky, for 3 detectors at points, and generically never for ≥ 4. B ~ ε² e^{2|z2|²/ε}.
  - (ii) Data peaked at pure circular polarisation, handled with Kirwin's generalised Laplace method with an integrable singular prior.

## Implementation details

- Use complex amplitudes per circular polarisation.
- Rotate to the dominant-polarisation frame, which diagonalises the amplitude metric.
- The phase integral gives I0, and its asymptotic form e^x/√(2πx) is valid for x ≳ 5.
- Replace the 1/R prior factor by its value at the Gaussian peak 2ρ/ζ. This is valid when 2ρ/ζ − R0 ≫ ζ^{−1/2}, i.e. for loud signals.
- There are no numerical experiments.

## Caveats

- Purely analytic. Ground-based long-wavelength response; the CW case is treated in the same way.
- There is no validation of the approximation error vs SNR (1311.0065 §VI does this for κ = 0).
