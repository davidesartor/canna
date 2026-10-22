# Global Analysis of the Gravitational Wave Signal from Galactic Binaries

Littenberg et al., 2020, PRD 102, 123021 · [arXiv:2004.08464](https://arxiv.org/abs/2004.08464)

**Evidence: B.** Peer-reviewed (Physical Review D); 74 citations in 6.5 yr (Semantic Scholar,
2026-10-19).

## What they do

GBMCMC.
- A time-evolving, trans-dimensional RJMCMC with parallel tempering over narrow frequency segments.
- N_GW ∈ U[0, 30] per segment, plus a piecewise-constant noise level per segment.
- Global proposals: an F-statistic 3-D histogram, sideband jumps, ψ–φ₀ mode jumps, and covariance proposals built from the previous epoch's posterior.
- Post-processing clusters samples into catalog entries by waveform match.

## What they find

- Fewer than 1% of SNR > 7 source pairs have match above 50% at 1 yr. That falls to 0.1% at 2 yr and 0 at 4 yr. This justifies proposing per source, as a blocked Gibbs sampler.
- The covariance proposal from earlier epochs cuts convergence time by orders of magnitude for a high-f₀, high-SNR source; without it, the T = 24-month run failed to converge.
- Demonstrated on a ~140 μHz LDC Radler band at 3.98 mHz, at 1.5 / 3 / 6 / 12 / 24 months. Results are qualitative; there is no P-P test.

## Implementation details

- **Parameters**: 8 per source, (A, f₀, ḟ, φ₀, ι, ψ, θ, φ).
  - A and ḟ are preferred over (Mc, d_L), so mass-transferring systems and poorly constrained ḟ need only one parameter. They resample to Mc and d_L in post-processing when ḟ > 0.
  - Priors: φ₀ ∈ [0, 2π], ψ ∈ [0, π], cos ι ∈ [−1, 1].
- **Amplitude prior**: placed on SNR, p(ρ) = 3ρ / (4ρ*² (1 + ρ/(4ρ*))⁵) with ρ* = 10. It vanishes at ρ → 0 and falls as ρ⁻⁴. This stops empty templates from saturating the model, BayesWave-style.
- **Sky prior**: galactic bulge plus disk, mixed with 10% uniform.
- **F-statistic proposal**: a 3-D (f₀, θ, φ) grid histogram of F, with spacing set by the reduced Fisher matrix. Extrinsics are drawn from the prior. Directed F-stat extrinsic jumps are listed as future work.
- **Degeneracy moves**:
  - Sideband jump f₀ → f₀ + n/yr, n = round(N(0, 1)).
  - **ψ–φ₀ jump (ψ ± δ/2, φ₀ ± δ), δ ~ U[0, 2π]**. It keeps φ₀ ∓ 2ψ fixed, i.e. it moves along the φ0 ± 2ψ directions. The sign of the correlation follows sign(cos ι).
  - **Exact two-fold mode**: (ψ + π/2, φ₀ + π). Every source has ≥ 2 modes in (ψ, φ₀).
- **Data handling**:
  - Only the A and E channels.
  - Segments are padded by about one source bandwidth. Samples fitting sources in the padding are dropped in post-processing.
  - Loud sources from the previous epoch are forward-modelled and subtracted before each segment run, then added back inside it.
- **Catalog**: samples are clustered with M* = 0.5 match within 10 bins. An entry's evidence is proportional to its sample count. Sources are labelled by their median f₀, which "is by far the best constrained parameter".

## Caveats

- Demonstration only, with no calibration statistics.
- Equal, constant arms; stationary noise; no gaps (all left as future work).
- Sources are taken to be uncorrelated in the proposals; the full likelihood still couples them.
