# F57 (2026-10-22): B-stft-hd — each source reads its band of the data Doppler-demodulated at its current sky

**Not in the current code.** The Doppler bands (`problem.DopplerGeometry`,
`network.doppler_bands`, `LisaFlow(doppler=...)`, `configs/B-stft-hd.yaml`,
`tests/lisa/test_doppler.py`) live on `origin/fml` at 6e2f01e and were not ported to the
restructured `canna.lisa`. Result: [F58](058-doppler-band-trims-typical-sky-offset.md).

**Why.** The current recipe's sky posteriors are individually biased (the user, from the corner
pages): B-stft-h up to 3 MCMC widths off at mid SNR (q0.10 s2: λ at −1.9 of the flow's own
width) and 11–12 on the loudest source; XS-stft-h up to 18–28 (q0.10 s2, s3). Among B-stft-h's
sources above SNR 15, 6 of 17 with a Doppler modulation index 2πf₀(1 AU/c)cos β ≥ 1 rad have λ or
β more than 1.5 flow widths off, against 1 of 17 below 1 rad. Above ~0.5 mHz the sky information
is mostly in the Doppler phase, and the network has to discover a reference to read it. The
template gave one (XS-stft-ht's loudest sky 3× narrower than XS-stft-h's) but cost f₀, because
it needs f₀ within a bin ([F54](054-xs-b-stft-ht-phases-learned.md),
[F56](056-phi0-learned-without-template.md)).

**The physics** (Cutler 1998, gr-qc/9703068; Cornish & Rubbo 2003, gr-qc/0209011; Cornish &
Larson 2002, gr-qc/0206017).
- Ψ(t) = 2πf t + φ₀ + φ_D(t) + φ_P(t), with φ_D(t) = 2πf n·r(t)/c, r(t) LISA's barycentre; in
  ecliptic angles 2πf (R/c) sin θ cos(2π f_m t − φ).
- Re-timing the data to the Sun's frame for one sky collapses a source's Doppler sidebands into a
  spike; the sharpest spike over the sky locates it.
- Doppler alone cannot tell north from south of the ecliptic, and ignores the antenna-pattern
  modulation.

**The change** (`network.doppler: true`; everything else is B-stft-h):
- **Orbit:** LISA's barycentre, as jaxgb samples it, fitted as centre + a cos(ωt) + b sin(ωt).
  Best period 365.26 d, ≤ 7×10⁶ m from the orbit file: ≤ 2×10⁻³ rad of Doppler phase at 12 mHz.
- **Per source:** the whitened A, E, T window goes to time, is multiplied by e^{−iφ_D(t)} at the
  source's current sky and f₀ (x_t), and back; the 16 bins around its f₀ are encoded like the
  STFT tokens through their own PLE and added to the source's token.
- **Checked:** on jaxgb signals at the true sky a 3–8 mHz source goes from 7–20% to 63–79% of its
  power in 5 bins, at the mirrored sky 8–11%. φ_D depends on f₀ only as a factor (~10⁻⁴ relative
  error for a bin), so unlike the template the band is right as soon as the sky is. Every other
  parameter's initialisation is unchanged.
- **Cost:** three 4096-point FFTs per source per network call.

**Predictions, written before the result** (against 26-lisa-B-stft-h):
1. Where the Doppler dominates (index ≥ 1 rad, SNR ≥ 15), at most 2 of 17 with λ or β more than
   1.5 flow widths off (now 6); sky width for q0.10 s0–s2 (index ≥ 1.4) ≤ 1.2× the MCMC (now
   1.4–2.2×).
2. Below index 1 rad, including the loudest source (0.16 mHz, index 0.5), within 10% of
   B-stft-h.
3. f₀ and the phases unchanged: per source within 10%, 215 ± 3 found, pinned φ± still 1.8–2.3×
   at mid SNR.
4. If the offsets do not move, the Doppler reference is not what the network lacks; next suspect
   the antenna-pattern part (amplitude and polarisation modulation), which needs ψ and ι.
