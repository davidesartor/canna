# Bayesian F-statistic-based parameter estimation of continuous gravitational waves from known pulsars

Ashok et al., 2024, PRD 109 104002 · [arXiv:2401.17025](https://arxiv.org/abs/2401.17025)

**Evidence: B.** Peer-reviewed (Physical Review D); 4 citations in 2.7 yr (Semantic Scholar,
2026-10-19).

## What they do

- Known pulsar, so the phase-evolution parameters λ are fixed. The 4-D amplitude posterior (h0, cos ι, ψ, φ0) is sampled with dynesty, using the exact Gaussian likelihood built from F-statistic by-products (F_a, F_b, A, B, C).
- **φ0 is marginalised analytically.** 𝒜^μ x_μ = q cos(φ0 − φ̄) and ρ² is independent of φ0, so L_φ0 = e^{−ρ²/2} I0(q). Here q² = 2h0²γ[α1|F_a|² + α2|F_b|² + 2α3 Re(F_a* F_b)].

## What they find

- Noiseless injection at ρ ≈ 100: parameters are recovered at the posterior peak.
- Symmetry: ψ → ψ + π/2 together with φ0 → φ0 + π leaves the signal unchanged. With the gauge ψ ∈ [−π/4, π/4), injections near the edge (ψ = −π/4) give a **bimodal ψ–φ0 posterior**. Dynesty finds both modes; BilbyMCMC with default settings finds only one (Figs. 2–3).
- P–P tests: 10 000 injections with log-uniform h0 ∈ [1e−28, 4e−26], ρ_max ≈ 50. Per-parameter KS p-values are fine except for h0 (89.2% inside the 90% CI). The φ0-marginalised version gives similar p-values.
- 4-D vs 3-D runtime: 154 s vs 124 s on 32 cores. The marginalised likelihood costs ~2× per call (µs scale).
- **Ultra-high SNR (ρ up to 1200): a systematic h0 bias appears in the P–P plot.** It is traced to a ≈3% error in the signal power ρ² from per-SFT approximations: linear phase within an SFT, constant antenna pattern, Dirichlet kernel truncated to Δk = 8 bins, noise estimation. "In the low-SNR regime, this 3% bias is absorbed in the width of the posteriors … but in ultrahigh SNR … systematic biases, even at a few per cent, begin to matter."
- Hardware injections: 12/17 are within 3σ; 1 is uninformative and 4 lie in the tails.

## Implementation details

- Gauge-fix ψ ∈ [−π/4, π/4) and φ0 ∈ [0, 2π), using the ψ + π, ψ + π/2 symmetries.
- Marginalise φ0 analytically with I0 to remove the bimodality and one dimension. Use the same quantities to cross-check the full-likelihood posteriors.
- Use a nested sampler for multimodal ψ–φ0.
- Validate with P–P plots plus KS p-values per parameter.

## Caveats

- Targeted CW only, with λ fixed (no f0/sky inference).
- Sampling-based, with no NPE.
- The ultra-high-SNR bias is a likelihood-approximation issue, not a sampler issue.
