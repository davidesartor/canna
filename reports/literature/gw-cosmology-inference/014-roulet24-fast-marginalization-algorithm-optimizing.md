# Fast marginalization algorithm for optimizing gravitational wave detection, parameter estimation and sky localization

Roulet et al., 2024, PRD 110 044010 · [arXiv:2404.02435](https://arxiv.org/abs/2404.02435)

**Evidence: B.** Peer-reviewed (Physical Review D); 13 citations in 2.5 yr (Semantic Scholar,
2026-10-19).

## What they do

factor the likelihood into intrinsic-only tensors and cheap extrinsic factors.
- Intrinsic-only tensors: the per-mode, per-polarisation, per-detector matched-filter time series z_mpd(t; θ_int) and covariances c_mm'pp'd(θ_int). These come from one waveform call.
- Extrinsic factors: antenna patterns F_dp(n̂, ψ), time shifts, e^{imφ_ref} and 1/D.
- Distance is marginalised by a 2-D lookup table (Singer–Price).
- Orbital phase is marginalised by trapezoid quadrature on a regular grid. It cannot be done analytically once higher modes are included, but the integrand is periodic, so trapezoid is spectrally accurate.
- (t⊕, n̂, ψ) are marginalised by adaptive multiple importance sampling with QMC. The output is the marginal likelihood plus weighted conditional samples of all extrinsics.

## What they find

- N_eff tracks the true error, with one power-law fit across events (Fig. 3).
- N_eff ≳ 100 gives about 10% precision on L, usually within 50 ms on one core (Fig. 4).
- GW190814: 1.6 h marginalised vs 5.5 h for the full 15-D run (3.5×). Other events were 3–5× faster, with matching posteriors.
- P–P plots (HLV, O3 PSD, ⟨h|h⟩ > 70, three chirp-mass bins, IMRPhenomXODE with precession and HM) are uniform. 1–2% of runs failed (24 h timeout, or posterior below the SNR cut).
- Runtime for 10⁴ effective samples: 0.78 h (high mass) to 2.6 h (low mass). Per-call marginalisation is 38–60 ms, about 70% of the total cost.
- Low-latency localisation of an NSBH at SNR 28.6: 4000 extrinsic samples in 12 s. Higher modes tighten inclination and therefore distance.
- Configuration (Table 1): Δ = 2⁻¹³ s, N_j = 2048 samples per proposal, N_eff,min = 50, up to 16 adaptations, N_φ = 128 phase quadrature points, ±70 ms time window.

## Implementation details

- (i) Only φ_ref is integrated at high resolution. ψ uses its uniform prior as the IS proposal, "rarely well constrained due to degeneracy with φ_ref".
- (ii) Phase samples are drawn from the quadrature-grid weights. Distance is drawn by inverse CDF.
- (iii) Sky proposal: precompute 10⁶ Halton-sequence sky points binned by discretised inter-detector delays. Propose discrete detector arrival times τ_d, then look up a sky point from that bin. The bin occupancy gives the physical delay prior Π(δτ).
- (iv) Per-detector proposals come from tempered (β ≤ 1) single-detector max-likelihoods. Detectors are sorted loudest-first, and each later detector's prior is the earlier proposal convolved with the delay prior.
- (v) Adaptation: a weighted KDE with a heavy-tailed Cauchy kernel, width from Silverman's rule floored at Δ, averaged with the previous proposal. Proposals are combined by the balance heuristic, and adaptation stops when N_eff reaches its threshold.
- (vi) Scrambled-Halton QMC on the inverse-CDF coordinates of the factorised proposal, which gives negatively correlated weights.
- (vii) A.1/A.2: relative binning extended with a time axis (multiple time-shifted references); sparse B-spline matrices.
- (viii) A.3: "stall" the reference waveform's high-frequency decay so the ratio h/h⁰ cannot blow up.
- (ix) A.4: time-align every template to the reference by a weighted LS fit to the unwrapped phase of h/h⁰. This is reported in the original convention for the user.
- (x) A.5: remember the proposal across calls (EMA with ε = 10⁻²).
- (xi) A.6: prune phase-grid points with Δ max_D ln L > 12.
- (xii) A.7, polarisation flip: when ⟨d|h⟩ < 0, ψ → ψ + π/2 flips the sign of h, which salvages anti-phase proposals.
- (xiii) A.8: stop refining low-likelihood points (more than 30 below the running max).
- (xiv) Inclination posterior "folding" for multimodality (ref [34]).

## Caveats

Gaussian noise, quasi-circular binaries. Inclination is intrinsic under precession. The likelihood is noisy at about 10%, which the sampler must tolerate. The localisation demo assumes the true intrinsics. This is ground-based detector machinery (arrival times), so the sky part does not transfer to LISA GBs.
