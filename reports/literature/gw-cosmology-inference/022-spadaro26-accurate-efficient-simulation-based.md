# Accurate and efficient simulation-based inference for massive black-hole binaries with LISA

Spadaro, Gair, Gerosa, Green, Buscicchio, Gupte, Tenorio, Clyne, Pürrer, Korsakova, 2026, arXiv v2 03 Aug 2026, astro-ph.HE · [arXiv:2603.20431](https://arxiv.org/abs/2603.20431)

**Evidence: B.** Peer-reviewed (Physical Review D); 0 citations in 0.6 yr (Semantic Scholar,
2026-10-19).

## What they do

Ports Dingo to LISA. An amortised NPE (RQ-spline coupling NSF, Dingo's standard architecture, 128-d embedding per channel, ~361M parameters in total) is followed by importance sampling with the exact likelihood (Dingo-IS).
heavy MBHBs (M ≳ 10⁷ M_⊙), so the low-frequency approximation (rigid, motionless constellation, A/E as two co-located Michelsons) applies, evaluated at a fixed reference time t_ref = 1 yr. IMRPhenomXHM (aligned spin, higher modes). Frequency domain 0.1–4 mHz, Δf = 10⁻⁵ Hz. SciRDv1 PSD, **no Galactic confusion noise**. 11 parameters: M_c, q, χ1,2, d_L, θ_JN, θ_S, φ_S, ψ, t_c, φ_c (SSB frame).
- **Priors / SNR:** m1, m2 ~ U[7×10⁶, 3×10⁷] M_⊙ with m1 > m2, i.e. M_c ∈ [8×10⁶, 2×10⁷], q ∈ [0.2, 1]. d_L ~ U[15, 50] Gpc ("improves training", following the Dingo prescription). Spin magnitudes U[0, 0.99] isotropic, so the aligned χ prior peaks at 0. t_c ~ U(±30 min) around a trigger. SNR ~10–1000.
- **Training:** 10⁷ intrinsic waveforms with extrinsics drawn on the fly. 400 epochs, batch 4096, Adam 2×10⁻⁴ with ReduceLROnPlateau (factor 0.4, patience 15). **~10 days on one A100-40GB.** Sampling: 2×10⁴ draws in <2 s, plus 10–15 s of IS on 20 CPUs.

## What they find

- **p-p:** 1000 injections pass KS on all 11 marginals (Fisher-combined p = 0.27); median IS efficiency ε ≈ 22%.
- **Comparison with Nessai** (4000 live points, 10–40 days per run):
  - SNR 87: ε = 50% (N_eff = 10⁴), multimodal sky recovered.
  - SNR 500: ε = 12%, small flow deviations fixed by IS.
  - SNR 1000: **ε = 0.05%, N_eff = 10**. The flow is broader than Nessai with little overlap, but still shrinks the prior volume by ~10¹³ at 90% CI.
- **JS divergence** over 100 noise-fixed injections:
  - low SNR: flow [6.3×10⁻⁴, 2×10⁻²], after IS [1.9×10⁻⁴, 10⁻²] bits;
  - moderate SNR: flow up to 0.1 bits, **with the worst being the phase parameter**; after IS ≤ 7.3×10⁻⁴.

## Implementation details

- Fix the detector orientation (t_ref) so the network does not learn the orbital modulation.
- Use a narrow t_c window from a trigger.
- Use a uniform-d_L training prior.
- Use IS for exactness, with ε as a free calibration diagnostic.

## Caveats

- **Efficiency is strongly anti-correlated with SNR** (2×10⁴ prior draws). They attribute this to the mass-covering forward-KL NPE objective, which makes the flow too diffuse when the true posterior is sharp.
- **Training longer did not help**; they suggest more capacity or a more flexible density instead.
- **Prior-coverage effects** also matter. The SNR-1000 source sits at M_c below the 3.5th percentile and χ1 above the 85th:
  - moving M_c to 1.3×10⁷ raises ε to 1.5%;
  - setting χ1 = 0 raises ε to 0.5%;
  - moving θ_S into a dense region does nothing.
- About 10 of the p-p injections with SNR 250–500 have ε < 1%; these sit at prior edges in M_c, χ1 or q.
- Sky and ψ remain broad and multimodal (an LFA degeneracy).
- **Label switching / multiple sources:** single source; no multimodality handling beyond the flow itself.
- Only the LFA and fixed orientation, which avoids the hard LISA response; no confusion noise. The criticism of Liang 2407.07125 (confusion noise in test but not training; systematic deviations) is stated by the authors.
