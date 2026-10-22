# Efficient Massive Black Hole Binary parameter estimation for LISA using Sequential Neural Likelihood

Martín Vílchez, Sopuerta, 2024, arXiv v3 24 Mar 2025, gr-qc · [arXiv:2406.00565](https://arxiv.org/abs/2406.00565)

**Evidence: B.** Peer-reviewed (Journal of Cosmology and Astroparticle Physics); 5 citations in
2.4 yr (Semantic Scholar, 2026-10-19).

## What they do

Targeted (non-amortised) SNL. A conditional MAF p_φ(x | θ) is trained on compressed data; posterior samples come from Eryn parallel-tempered MCMC on p_φ(x_o | θ)p(θ), and those samples seed the next round of simulations. The data are compressed in a separate pre-step, because a jointly trained embedding would collapse (the NLE objective is minimised by a summary independent of θ). Two compressors: PCA-128 of whitened time series (98.9% explained variance clean, 98.7% noisy) and a 1D-CNN (denoising) autoencoder to 128-d on per-time-standardised data.
one MBHB (LDC2b Spritz source 1, SNR > 4000), IMRPhenomD via lisabeta, Keplerian orbits, TDI-2 A/E, 30 h window (−24 h / +6 h) at Δt = 15 s → 2 × 7200 samples. Stationary Gaussian noise added after whitening. 11 parameters in the LISA frame: λ_L ∈ U[0, 2π], sin β_L ∈ U[−1, 1], χ1,2 ∈ U[−1, 1], Δt_c ∈ U[−10³, 10³] s around a presumed search estimate, φ_c ∈ U[0, 2π], cos ι ∈ U[−1, 1], ψ_L ∈ U[0, 2π], D_L ∈ U[3, 100] Gpc, M_c ∈ U[10⁵, 10⁷] M_⊙, q ∈ U[0.125, 1].

## What they find

Qualitative only (corner plots, no numbers or p-p). SNL-PCA recovers intrinsics with widths similar to Eryn, slightly broader with noise, and m₁–m₂ correlations are lost. λ_L stayed at the prior until a sudden "breakthrough" at round 33. SNL-AE is broader and biased in sky/ι, because the AE drops the merger — the effective data sit in the low-frequency regime, so the antipodal/reflected sky modes of Marsat et al. appear (up to 6 of the 8 predicted modes).

## Implementation details

Sbi MAF, 15 MADEs (2 × 128 hidden), with a shallow 128-unit θ-embedding trained jointly. 100 rounds × 10⁴ simulations (10⁶ waveform calls in total, vs 6.4×10⁷ for the Eryn reference). About 80 h on a 12-core Ryzen 5900X (no GPU benefit), vs 6 h for direct Eryn. The surrogate likelihood costs 3 ms vs 4 ms for the true one; 5 MADEs instead of 15 is 3× faster at no visible loss.

## Caveats

**ψ and φ_c are never determined by SNL** (Eryn finds them bimodal); it is unclear whether compression loss or too few rounds is to blame. The AE cannot reconstruct the merger (most of the SNR). Fully-connected AEs overfit. Gibbs-like slice sampling mixes poorly on correlated parameters, so they used Eryn instead. Training slows as the dataset grows, and the number of simulations per round trades speed against final quality.
- **Label switching / multiple sources:** single source; the method is per-observation.
- One injection, the same simulator for train and test, no quantitative comparison or calibration, and much slower than direct MCMC in wall time.
