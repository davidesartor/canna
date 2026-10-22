# Fast and accurate parameter estimation of high-redshift sources with the Einstein Telescope

Santoliquido et al., 2025, PRD 112 103015 · [arXiv:2504.21087](https://arxiv.org/abs/2504.21087)

**Evidence: B.** Peer-reviewed (Physical Review D); 5 citations in 1.5 yr (Semantic Scholar,
2026-10-19).

## What they do

apply the DINGO NPE + importance sampling pipeline unchanged to the triangular ET.
- Setup: high-z BBH, IMRPhenomXPHM with aligned spins, 8 s segments, 6–256 Hz.
- Validated against Bilby/nessai, then compared with Fisher (GWFish+Priors) on 1000 injections.

## What they find

- Event 1 (ρ = 83, z = 2.5):
  - Sample efficiency 16% (5×10³ effective samples from 3×10⁴).
  - log Z = −6073.89 ± 0.01 (Dingo-IS) vs −6073.58 ± 0.08 (Bilby).
  - All 1-D JSDs are below a bootstrap 95th-percentile threshold (e.g. ϕ_c ⟨JSD⟩ 0.4 vs threshold 1.4, in 10⁻³ nat).
  - The weight plot shows a tail with high q and low p only, i.e. q is slightly too wide, which is the desired direction.
- P–P plot for 1000 injections without IS: combined p = 0.001. "Most notably" the deviation is in the phase ϕ_c. IS is therefore needed.
- Sample efficiency: mean 12.5%, median 10%; 85% of events above 1%; about 18% at z > 4 and 20% at z > 10. Efficiency is lowest at z ≲ 2, i.e. at high SNR.
- Pop III catalogue (690 events): mean 11.4%, median 8.1%, 75% above 1%.
- Fisher vs NPE:
  - Fisher underestimates the sky area by about 8× for well-localised events, and by up to 10³× for poorly localised ones.
  - Fisher overestimates the d_L error by 2–4× at d_L > 10⁵ Mpc.
  - Fisher underestimates the M_c, q and χ errors.
  - 156 of 500 events had a near-singular Fisher matrix.
- Higher modes vs 22-only in Bilby: 10 h (3.7×10⁷ likelihood calls) vs 13.75 days (8.5×10⁸). Known precession reduces 8 sky modes to 4.
- Cost: training 6 days on 1 A100 + 32 CPUs (970 kWh); inference 1.7 min per event on 8 CPUs.

## Implementation details

- **ϕ_c is not learned:** "we also analytically sample the phase parameter ϕ_c after training rather than trying to learn it… does not require any approximations (Dax et al. 2023)". The flow models all other parameters; ϕ_c is drawn from its exact conditional given the rest, then IS weights are computed.
- GNPE and PSD conditioning were skipped for simplicity.
- Network: embedding to 128 features plus a flow, >3×10⁸ parameters, 5×10⁶ waveforms, LR plateau scheduler with patience 10.
- Validation: bootstrap JSD thresholds (100 sets × 10⁴ samples, 4950 pairs, 95th percentile).
- Events with efficiency below 10% are dropped from science plots, which is a selection effect.
- Explicit symmetry list (Table 3): β → −β, λ → λ + π, and (λ + π/2, ψ + π/2) give 8 modes. The same structure appears for LISA MBHBs under the long-wavelength, static-antenna approximation (Marsat+21).

## Caveats

- Fixed PSD, Gaussian noise, aligned spins, short signals only.
- Selection on sample efficiency.
- Even in the paper with exact phase handling, the uncorrected NPE P–P is worst for the phase. The analytic ϕ_c step happens before IS; the reported P–P is for the raw Dingo proposal.
