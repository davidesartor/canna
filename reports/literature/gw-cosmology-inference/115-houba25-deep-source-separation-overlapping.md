# Deep source separation of overlapping gravitational-wave signals and non-stationary noise artifacts

Houba, 2025, arXiv v2 27 Jun 2025, astro-ph.IM · [arXiv:2503.10398](https://arxiv.org/abs/2503.10398)

**Evidence: C.** Not peer-reviewed; 0 citations in 1.6 yr (Semantic Scholar, 2026-10-19). Its
claims are the authors' alone: a hypothesis, not a result.

## What they do

Demucs-inspired (no code reuse, no skip connections, no BLSTM) time-domain encoder → shared bottleneck → three decoder heads: MBHB (1 channel; assumes no MBHB overlap in time), glitch (1 channel, all glitches lumped), GB (**N_GB = K frequency-bin channels**, each reconstructing the GB content whose centre frequency falls in that bin). Encoder 4 Conv1D (1→64→128→256→512, k8 s2 ×3), bottleneck Conv1D 512→256 (channel reduction only), decoders 3 ConvTranspose1D. 2.79 M params. Each TDI channel (A, E, T) processed independently by the same model (cross-channel correlations discarded).
- **Label switching / multiplicity:** avoided by construction — GB slots are fixed **frequency bins** (one source per bin assumed), so no permutation ambiguity and variable N via silent channels. Authors note this breaks in dense spectra and propose dynamic multi-slot decoding with Slot Attention and Hungarian-matched supervision (the SlotFlow follow-up).
- **Data / training:** BBHx (IMRPhenomHM, masses U(10⁵,10⁶) M⊙, z 2–5 uniform in comoving volume, aligned spins) + FastGB (f log-U 1–10 mHz, A log-U 10⁻²²–2×10⁻²¹, up to 5 GBs, isotropic sky), 0–30 Gaussian-burst glitches, stationary LISA noise; Δt = 5 s, **2-hour** snippets; K = 5 bins of 1.8 mHz. 25,000 training samples, Adam 1e-3, batch 16, 25 epochs, ~4 h on a MacBook M2 Max. Loss = sum of unweighted MSEs.

## What they find

MBHB median match > 0.95 above SNR ≈ 15; GB median match > 0.95 above SNR ≈ 10, but **declines at high SNR** (attributed to OOD). Glitch-timing sweep at SNR 50: match ~flat, slight drop when the glitch overlaps the merger. MBHB-decoder hallucination area 6.88×10⁻⁵ on quiet data. Count of active GBs (1 or 5) recovered in examples. Downstream toy SBI (sbi NSF, 50k sims, sky λ, β only): posterior from a low-match reconstruction drifts in longitude; retraining the SBI on separator outputs ("separator-aware") restores agreement with the true-injection posterior.

## Implementation details

Bin-indexed output slots to sidestep permutation; train on short random segments and stitch with overlap at inference (planned); separator-aware SBI.

## Caveats

2 h segments give ~0.14 mHz resolution, so GBs are short sinusoids, not Doppler-modulated year-long signals; 5 bins for ≤ 5 GBs means overlap is essentially excluded; equal-weight MSE lets loud MBHBs dominate and underfit GBs; silent channels not exactly zero; no posteriors on GB parameters; no comparison with any likelihood-based method.
