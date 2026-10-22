# Removing degeneracy and multimodality in gravitational wave source parameters

Roulet, Olsen et al., 2022, arXiv v2; cogwheel paper · [arXiv:2207.03508](https://arxiv.org/abs/2207.03508)

**Evidence: B.** Peer-reviewed (Physical Review D); 26 citations in 4.3 yr (Semantic Scholar,
2026-10-19).

## What they do

design invertible coordinates with simple Jacobians. Each well-measured observable (amplitude, phase and arrival time at the loudest detector, and the time delay to the second detector) is controlled by one coordinate. The poorly measured combinations go into separate coordinates. Four approximate discrete symmetries then make the residual multimodality removable by **folding**.

## What they find

- **Multimodality:** up to 2⁴ = 16 modes; folding reduces the number of disjoint modes by up to 8×.
- **Mode finding on GW151226 (PyMultiNest, 512–16384 live points):** without the new coordinates and folding, only the 16384-live-point run finds all modes, and even then one is undersampled. Every folded run finds all modes.
- **Accuracy:** the Jensen–Shannon error on (θ_JN, φ_net) is more than 10× lower at equal runtime.
- **Symmetry check:** unfolding probabilities cluster near 1/16 (GW151012), which confirms the symmetries are good approximately in the full 15-D space.
- **Expected precision:** Δa/a ~ 1/ρ, Δφ ~ 1/ρ, Δt ~ 1/(2πσ_f ρ). These are matched by the folded posterior, except t_k0, which correlates with the intrinsics.
- **Practical tricks / implementation details** (the actual transforms, Table 1):
- **Masses:** (ℳ, ln q) instead of (m₁, m₂).
- **Chirp distance d̂ ∝ d_L / (ℳ^{5/6}|R_k0(ι, n̂, ψ)|).** This removes the d_L–ι–sky correlation and makes the observable range mass-independent. Jacobian d̂/d_L. Optionally marginalise distance semi-analytically.
- **Arrival time at the reference detector,** t_k0 = t_c − n̂·r_k0/c, instead of geocentric t_c. The ~40 ms sky correlation shrinks to ≲1 ms. Unit Jacobian.
- **Sky in the detector-pair frame (θ_net, φ_net):** the z-axis runs through the two loudest detectors, so cos θ_net is fixed by the time delay and well measured. The isotropic prior is uniform in (cos θ_net, φ_net).
- **Arrival-phase coordinate φ̂_ref:** φ_ref shifted by the response phase arg R_k0 and by 2π f̄ t_k0. It is defined relative to the max-likelihood arrival phase, so its posterior peaks near 0 with width ~1/(2ρ). The time-phase term makes it orthogonal to t_k0. Unit Jacobian.
- **Reference frequency f_ref = f̄_k0** (the SNR²-weighted mean frequency): define evolving parameters in band to avoid spurious correlations.
- **Spins:**
  - χ_eff plus the prior-CDF of the aligned-spin difference; uniform priors, so no Jacobian is needed.
  - Spin azimuths relative to the line of sight N̂, not the orbital separation (θ_JN, φ_JL, φ₁₂).
  - In-plane magnitudes as prior CDFs C^⊥ ∈ (0, 1).
- **The four approximate symmetries:**
  - φ̂_ref → φ̂_ref + π (exact for even m).
  - ψ → ψ + π/2 at fixed φ̂_ref.
  - φ̂_net → −φ̂_net.
  - cos θ_JN → −cos θ_JN.
  - The coordinates are deliberately redefined (φ̂_net, φ̂_JL with conditional π-shifts) so that each symmetry is a **one-parameter shift or reflection**.
- **Folding:** sample P̃(θ̃) = Σ_i P(σ_i(θ̃)) over the 16 sector maps, then assign each sample to sector i with probability ∝ P(σ_i θ̃).
  - It is exact even when the symmetry is broken.
  - It costs ≈ one likelihood evaluation, because only extrinsics change and waveforms are reused: e^{imφ_ref} phases and antenna factors.
  - Folding also removes branch-cut discontinuities, which sit on the fold.

## Caveats

- The symmetries are specific to quadrupole-dominated signals and the nearly co-aligned HL network. With Virgo loud, or strong higher modes, they break, and folding is then harmless but useless.
- The reference phase needs a max-likelihood configuration found first.
- The paper has no neural component.
