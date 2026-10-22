# Rapid Parameter Estimation for Merging Massive Black Hole Binaries Using Continuous Normalizing Flows

Liang, Du, Wang, Xu, Liu, Wei, Xu, Qiang, Luo, 2024, arXiv v3 05 Dec 2024, gr-qc; Mach. Learn.: Sci. Technol. 5, 045040 (2024) · [arXiv:2407.07125](https://arxiv.org/abs/2407.07125)

**Evidence: B.** Peer-reviewed (Machine Learning: Science and Technology); 8 citations in 2.3
yr (Semantic Scholar, 2026-10-19).

## What they do

FMPE (Dax et al.: Gaussian path, mean tθ₁, std 1−(1−σ)t, so the base is N(0, I)) and a "VPFMPE" variant with the trig interpolant sin(πt/2)θ₁ + cos(πt/2)θ₀ at fixed σ. Both are applied to the Du et al. 2308.05510 Taiji MBHB setup, reusing its frame-transform generalisation trick (train at t_ref = day 1, map samples through the analytic SSB↔detector rotation O₂(t_ref) for {λ, β, ψ}, Jacobian-corrected). New network "TResMlp": a sinusoidal Fourier embedding of t (learned frequency vector), a linear data embedding of the two TDI channels, and a residual MLP, replacing DINGO's DeResNet + GLU conditioning on (θ, t).
same 11 parameters as Du but with the full spin range: M_c ∈ [10⁵, 10⁶] M_⊙, q ∈ [0.1, 1], χ ∈ [−1, 1], t_c ± 0.01 d, d_L ∈ [6, 100] Gpc, angles uniform or isotropic. IMRPhenomD (2,2), BBHx with Taiji equal-arm orbit, TDI A/E, 1 day at 1/15 Hz. The ASD is also given as context. Training uses instrument noise only; test adds the GB foreground from ~3×10⁷ GBs.

## What they find

Table 3 (850 injections; Pearson correlation of posterior median vs truth for M_c / q / χ1 / χ2 / d_L):
- VPFMPE-TResMlp: 0.998 / 0.979 / 0.972 / 0.816 / 0.941 (189M params, 2.0 d)
- FMPE-TResMlp: 0.998 / 0.979 / 0.972 / 0.814 / 0.940 (189M, 2.0 d)
- FMPE-DeResNet: 0.997 / 0.976 / 0.956 / 0.717 / 0.940 (211M, 2.3 d)
- NPE-NSF-DeResNet: 0.926 / 0.865 / 0.662 / 0.168 / 0.863 (384M, 6.5 d)
- Training: 5×10⁶ signals, 400 epochs, batch 14336, Adam with cosine schedule, one A800. Figure 5: p-p over 4000 injections (t_ref 1–365 d) fails without the frame trick and passes with it. Single-event check vs nested sampling, SNR 1543 (the same injection as Du et al.), 1σ widths: M_c ±1.2×10³ vs ±115 (~10×); q ±0.023 vs ±0.0065 (~3.5×); d_L ±1.5×10³ vs ±890 Mpc (~1.7×); ι ±0.02 vs ±0.01 (2×); λ ±0.032 vs ±0.018 (~1.8×); β offset by ~5σ_Nest. Sampling: 14336 samples in 240 s on GPU vs 3000 s for nested sampling started near the truth.

## Caveats

Confusion noise (absent in training) degrades β and spins ("within 2σ or 3σ"); the medians of t_c, φ_c, ι, β, λ, ψ are not usable because of multimodality; NPE with discrete flows distorts spins badly; the linear vs trig path makes essentially no difference (correlations agree to 3 decimals).
- **Label switching / multiple sources:** single source.
- The "comparable to nested sampling" claim rests on one event, and intrinsic widths are still up to ~10× broader. Correlation coefficients are a weak metric (no p-p under confusion noise, no width ratios across injections). The nested sampling was started near the truth.
