# Advancing space-based gravitational wave astronomy: Rapid parameter estimation via normalizing flows

Du, Liang, Wang, Xu, Luo, Wu, 2023, arXiv v2 20 Feb 2024, astro-ph.IM; Sci. China PMA 67, 230412 (2024) · [arXiv:2308.05510](https://arxiv.org/abs/2308.05510)

**Evidence: B.** Peer-reviewed (Science China Physics Mechanics and Astronomy); 16 citations in
3.2 yr (Semantic Scholar, 2026-10-19).

## What they do

DINGO-style NPE for a single Taiji MBHB merger. An embedding network (per-channel linear compression of [Re, Im, ASD] × 2877 bins → 400 each, concatenated to 800, then a 32-block ResNet 1024→128) feeds an nflows RQ-coupling NSF (30 flow steps, 5 transform blocks, hidden 1024, 8 bins, ELU, batch norm), trained by NLL. The PSD is passed as context (ASD channel) so the network can take "off-source" noise estimates. **Time-translation trick:** training uses t_c within ±0.01 d of one reference day (day 30); at inference the data are phase-shifted e^{2πif(t'−t)} to day 30, sampled, then the posterior samples are mapped SSB→detector frame at t_ref and back detector→SSB at t'_ref. This exploits the near-static detector over 1 day to cover a whole year with one network.
11 parameters (M_c, q, χ1z, χ2z, t_c, φ_c, d_L, ι, ψ, λ, β), all uniform boxes (Table 1): M_c ∈ [2.5×10⁵, 1.25×10⁶] M_⊙, q ∈ [0.125, 1], spins in [0, 0.99] (positive only — a reviewer flagged this as bias-prone), d_L ∈ [6, 100] Gpc, ι and β isotropic. Inputs: IMRPhenomD (2,2) only, FD fast response, equal-arm analytic orbits, TDI-1 A/E, 1-day window at 1/15 Hz. Training SNR spans 10²–10⁴; the test event has SNR 1543. Training uses stationary instrument noise only; the test adds GB confusion noise (LDC catalogue of ~3×10⁷ GBs on Taiji orbits, with ~2×10⁴ sources of SNR > 7 removed).

## What they find

p-p over 1000 injections (confusion noise, t_ref 1–365 d) lies within the bands, with KS p-values in the legend. One-event comparison vs nested sampling (90% CIs): M_c ±3.7×10³ (NF) vs ±~100 (Nest); d_L ±5–6 Gpc vs ±0.7; ι ±0.1 vs ±0.008; λ ±0.3 vs ±0.014. The NF posteriors are **~an order of magnitude (up to ~40×) broader** than nested sampling; the authors call this "adequate when compared to the prior". Speed: 10⁴ samples in 2.7 s vs 3000 s on 90 CPU cores. Training: 5×10⁶ stored waveforms (d_L and t_c resampled on the fly), 600 epochs, batch 10240, Adam with cosine schedule from 5×10⁻⁵, about 6 days on one A800.

## Caveats

A **spurious second t_c mode** absent from nested sampling, which propagates into φ_c, ψ, λ, β ("the NF may not be powerful enough"). φ_c is not recovered (truth outside the 90% CI) — the flow cannot capture the intrinsic multimodality from (2,2)-only. No evidence computation.
- **Label switching / multiple sources:** single source; overlapping MBHBs are left as future work.
- A single nested-sampling comparison; p-p passes even though the posteriors are 10–40× too broad. This is the classic case where calibration ≠ sharpness, because the p-p checks marginal coverage only and is satisfied by over-dispersion against prior-ish truths. Positive spins only; no confusion noise in training.
