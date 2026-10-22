# FluxMC: Rapid and High-Fidelity Inference for Space-Based Gravitational-Wave Observations

Liang, Liu, Song, Zhao, Du, Wang, Gu, He, Xu, et al., 2026, arXiv v2 08 Apr 2026, astro-ph.IM · [arXiv:2604.04957](https://arxiv.org/abs/2604.04957)

**Evidence: C.** Not peer-reviewed; 0 citations in 0.5 yr (Semantic Scholar, 2026-10-19). Its
claims are the authors' alone: a hypothesis, not a result.

## What they do

An amortised FMPE (linear OT path, Gaussian base) gives a fast global approximation. Seeding PTMCMC from it skips burn-in and avoids local traps, while the MH correction keeps the result asymptotically exact.
single MBHB, LISA or Taiji (A2/E2 TDI-2, full time/frequency-dependent response via t_f(f)). IMRPhenomD or IMRPhenomHM (6 modes). Frequency domain, whitened by √S_n·√(T/4) plus fresh complex N(0,1) noise every step ("infinite training horizon", GPU on-the-fly generation).
- 11 parameters: log M_c ∈ [10^5.5, 10^6.5] (log-uniform), q ∈ [0.1, 0.9], χ_z1,2 ∈ [−0.9, 0.9], t_c ± 0.01 day, φ_c ∈ [0, 2π], D_L ∈ [10⁴, 10⁵] Mpc (log-uniform), cos ι, λ, sin β, ψ ∈ [0, π].
- SNR not stated. Data span 5 days in Case I.

## What they find

- IMRPhenomD, well-behaved event: FluxMC 0.42 h vs PTMCMC 28 h (67×).
- Cases I/II: PTMCMC is confidently biased in **φ_c and ψ** (and sky); FluxMC is unbiased, 1.7 h vs 75 h and 0.41 h vs 11.3 h.
- IMRPhenomHM: full-prior PTMCMC on an A100 for "hundreds of hours" lands in spurious q/χ modes. FluxMC takes ~4.5 h and matches a "near-truth" PTMCMC reference. Mean JSD vs that reference over 11 parameters:
  - Case III: 0.726 (PTMCMC) vs 0.019 (FluxMC);
  - Case IV: 0.581 vs 0.003.
- p-p: FluxMC KS p = 0.737 vs PTMCMC p = 0.033 — on 40 injections (20 D, 20 HM) in the appendix, but "30" in the main text.
- Higher modes remove the φ_c multimodality seen with IMRPhenomD.

## Implementation details

A dense-residual-block conditioner compresses the strain to a context c; sinusoidal t-embedding; concatenated with θ_t into a ResMLP. **No sizes, training-set size, epochs or GPU-hours reported.**

## Caveats

Framed entirely as PTMCMC failures. Phase/polarisation (φ_c, ψ) are the parameters where local samplers are trapped. No failures of the flow itself are reported.
- **Label switching / multiple sources:** single source.

- The reference is a PTMCMC initialised at the truth, so it is not independent of mode-finding.
- No network or training details, and the injection count is inconsistent (30 vs 40).
- The "PTMCMC fails after hundreds of hours" baseline is a strawman given known heterodyned/GPU samplers.
- The JSD numbers mostly measure the baseline's failure, not the flow's accuracy.
- The flow-only posterior quality is never shown.
