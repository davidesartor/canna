# New Coordinates for the Amplitude Parameter Space of Continuous Gravitational Waves

Whelan et al., 2014, CQG 31 065002 · [arXiv:1311.0065](https://arxiv.org/abs/1311.0065)

**Evidence: B.** Peer-reviewed (Classical and Quantum Gravity); 21 citations in 13.0 yr
(Semantic Scholar, 2026-10-19).

## What they do

- The signal is linear in the 4 JKS amplitudes 𝒜^μ(h0, χ = cos ι, ψ, φ0). The paper takes linear combinations of these to get "CPF" (circular-polarisation-factored) coordinates.
- The CPF coordinates split into a right-circular pair and a left-circular pair: Ă¹ + iĂ² = A_r e^{iφ_r} and Ă³ + iĂ⁴ = A_l e^{iφ_l}.
- Here A_r = h0((1+χ)/2)², A_l = h0((1−χ)/2)², φ_r = φ0 + 2ψ and φ_l = φ0 − 2ψ. These two angles are the phase combinations φ0 ± 2ψ.
- The inverse map is h0 = (√A_r + √A_l)² and χ = (√A_r − √A_l)/(√A_r + √A_l).
- "Root-radius" coordinates: r = A^{1/4}, with x_r + i y_r = r_r e^{iφ_r} (and the same for l). The Jacobian from these to (h0, χ, ψ, φ0) is constant: dh0 dχ dψ dφ0 = 4 dx_r dy_r dx_l dy_l. So a prior that is uniform in h0, isotropic in ι and ψ, and uniform in φ0 is flat in root-radius space.
- The paper uses these coordinates to evaluate the B-statistic, i.e. the likelihood marginalised over the amplitudes under a physical prior.

## What they find

- Jacobian: d⁴𝒜 = 16 (h0(1−χ²)/4)³ dh0 dχ dψ dφ0, and dφ_r dφ_l = 4 dψ dφ0.
- The quadratic part of the log-likelihood in CPF-polar coordinates is I A_r² + J A_l² + 2 A_r A_l [K sin(φ_r − φ_l) + L cos(φ_r − φ_l)].
  - It depends on the angles only through φ_r − φ_l = 4ψ. **φ0 enters only through the linear term (the data projection).**
  - In the long-wavelength / diagonal-metric limit, I = J and the cross term vanishes, so the right and left blocks decouple.
- Circular polarisation (χ = ±1) is the origin of one polar system, so only one circular phase is constrained there.
- Gaussian (Laplace) approximation of the B-integral:
  - In CPF coordinates it fails: the measure ∝ (A_r A_l)^{-3/2} turns the expansion into a saddle when Â_r/h_det < √(3/2) ≈ 1.22. A true local maximum exists only when Â_r/h_det > √6 ≈ 2.45.
  - In root-radius coordinates it gives ln B ≈ F − (3/2) ln(Â_r Â_l) + const. This is valid for strong signals that are not near circular polarisation.
- Exact results:
  - In physical coordinates the φ0 integral is analytic (2π I0) and the h0 integral is analytic for a uniform h0 prior, leaving a 2-D integral over (χ, ψ).
  - With a diagonal metric (A = B), the integral is exact in closed form via the confluent hypergeometric function ₁F₁(1/4, 1, Â_r²/2h_det²). Fig. 4 shows the Gaussian approximation converging to it only at large Â_r.

## Implementation details

- (i) Parametrise the 4 amplitude dof as two complex numbers, one per circular polarisation. Each phase pairs with its own radius, so the phase is undefined only where that radius → 0.
- (ii) Use r = A^{1/4} Cartesian coordinates to get a flat, singularity-free measure for the physical prior. This is a natural chart for a flow or NPE.
- (iii) φ0 can be marginalised analytically (Bessel I0).
- (iv) Laplace approximations in polar or JKS coordinates break down for weak or near-circular signals.

## Caveats

- Written for continuous waves from ground-based detectors (SFT inner products). The closed forms assume a diagonal amplitude metric.
- There is no numerical sampling or NPE experiment. The results are analytic only.
- The LISA GB F-statistic has the same 4-amplitude structure, but with LISA response functions.
