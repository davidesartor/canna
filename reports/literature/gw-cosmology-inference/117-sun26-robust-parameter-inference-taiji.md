# Robust parameter inference for Taiji via time-frequency contrastive learning and normalizing flows

Sun, Liang, Song, Liu, Jin, Wang, Du, et al., 2026, arXiv v1 15 Apr 2026, gr-qc · [arXiv:2604.13867](https://arxiv.org/abs/2604.13867)

**Evidence: C.** Not peer-reviewed; 0 citations in 0.5 yr (Semantic Scholar, 2026-10-19). Its
claims are the authors' alone: a hypothesis, not a result.

## What they do

Amortised NPE for MBHBs in data with glitches. An RQ-spline coupling NSF is conditioned on a gated fusion of a time-domain CNN branch and an FFT (Re/Im) CNN branch. An InfoNCE contrastive loss treats two noise+glitch realisations of the same h(θ) as a positive pair, making the context glitch-invariant. A neural glitch generator (conditioned on dv, τ1, τ2) replaces the physical glitch simulation (67.6 h → 2 min per 10⁵ samples on 100 CPU cores, 795×).
Taiji, IMRPhenomT, 11 parameters, with priors identical to FluxMC 2604.04957 Table 2: log M_c ∈ [10^5.5, 10^6.5], q ∈ [0.1, 0.9], χ ∈ [−0.9, 0.9], t_c ± 0.01 d, φ_c, log D_L ∈ [10⁴, 10⁵] Mpc, cos ι, λ, sin β, ψ ∈ [0, π]. LPF-style acceleration glitches with dv ∈ [10⁻¹⁵, 10⁻¹³] and τ1,2 ∈ [2, 60] s, rescaled to glitch SNR 8–32 in training (stronger in test). MBHB SNR range not stated.
- **Training:** 10⁵ fresh samples per epoch (Latin hypercube in θ, 90/10 train/val). Epochs, network sizes and GPU time are not reported. Inference: 3.2×10⁴ samples in 0.6 s vs 23 min for MCMC (RTX A6000).

## What they find

- One injection with a glitch (Table 3, medians ± 1σ). The Whittle-likelihood MCMC is biased: q 0.37 vs true 0.28, χ1 −0.87 vs −0.49, sin β 0.66 vs 0.36. NPE is centred, but far broader:
  - log M_c ±1.5×10⁻² vs MCMC ~10⁻⁴ (~35–100× wider);
  - **φ_c ±2.1 rad (≈ the uniform prior on [0, 2π]), ψ ±1.0 rad on [0, π] (≈ prior)**;
  - λ (−1.0, +3.9).
- p-p: all methods are calibrated without glitches. With glitches, MCMC fails while the full model passes; dropping the contrastive term degrades it strongly, frequency-only is worst, time-only is competitive.
- The normalised CRPS stays flat versus glitch duration and merger-relative timing for the full model.

## Caveats

Ablations (no contrastive loss, single domain) are less robust. **Appendix A:** marginal p-p/KS tests admit a broad valley of shifted or rescaled (too wide / too narrow) posteriors that still pass. They therefore add CRPS, which has a unique minimum at Δμ = 0, α = 1.
- **Label switching / multiple sources:** single source plus a single glitch.

- The "better than MCMC" claim compares against a deliberately misspecified likelihood; there is no glitch-aware MCMC reference.
- The NPE posterior is much wider than the MCMC posterior; this is passed off as calibration.
- No details on training size or cost.
- φ_c/ψ are essentially not learned (prior-wide).
