# Neural posterior estimation of Galactic Binary signals for the LISA mission

Delmond et al., 2026, arXiv preprint, astro-ph.IM · [arXiv:2606.29039](https://arxiv.org/abs/2606.29039)

**Evidence: C.** Not peer-reviewed; 0 citations in 0.3 yr (Semantic Scholar, 2026-10-19). Its
claims are the authors' alone: a hypothesis, not a result.

## What they do

- Conditional neural spline flow NPE, built on LISAFlow/nflows: 12 RQ-spline coupling steps, K = 10 bins, 10-block ReLU ResNet conditioners. Base N(0, I₈).
- Trained with standard −log q on 8 parameters (f0, ḟ0, λ, β, A, ι, φ0, ψ), normalised by prior mean and std. Raw concatenated Re/Im A,E bins are the input, with **no embedding network**.
- Narrow 4 µHz f0 prior inside a 6 µHz window around a fixed f_c. One model per f_c.
- Signals from GBGPU/FastGB, first-generation TDI, 1 yr observation. Noise drawn per bin as N(0, S_n T/4) on each of Re/Im. Only sources with SNR 10–100 are kept.
- Training data use a "hybrid on-the-fly" JKS trick: 7×10⁶ precomputed basis quadruplets (Ã_µ, Ẽ_µ)(f0, ḟ0, λ, β). The amplitude parameters (A, ι, φ0, ψ) are redrawn every batch and combined linearly via 𝒜^µ, with noise added on the fly.

## What they find

- Model and cost:
  - 64 bins per channel at f_c ≤ 6 mHz (512 neurons, 1.2×10⁸ params). Up to 512 bins at 10–15 mHz (1024 neurons, 3.9×10⁸ params).
  - About **one week per model on one A100**. Batch 2048, lr 2e-4 with ReduceLROnPlateau (×0.5 after 10 stale epochs). No overfitting observed.
  - 10⁴ samples in under 1 s vs about 1 h for ptemcee parallel-tempering MCMC.
- Low frequency:
  - At SNR 11 (1 mHz) the flow matches MCMC, including the multimodal φ0/ψ structure.
  - At SNR 60 (5 mHz) the f0, ḟ0, β, λ marginals are slightly broader than MCMC.
  - p-p plot over 1000 injections at 1 mHz: KS does not reject uniformity at 95%.
- JS divergence vs MCMC over 100 injections: median < 0.05 at 1–6 mHz and 0.05–0.1 at 10–15 mHz. Phase parameters (f0, ḟ0, λ, β) are **systematically broader** than MCMC at high f.
- Wider band (f_c ~ U[4.0, 4.5] mHz, f_c concatenated to the input):
  - The amplitude block (A, ι, φ0, ψ) is unchanged.
  - Sky JS is worse, and the f0 posterior is systematically broader than MCMC.
  - The authors blame the prior/posterior width ratio: a 0.5 mHz prior vs a ~10⁻³ µHz posterior width.
- Two sources at 5 mHz, 18 flow steps, 1.7×10⁸ params:
  - Label switching appeared as two modes, so sources were **ordered by f0** (f01 ≤ f02).
  - Results vary widely. With Δf0 = 0.5 µHz (partial overlap) the flow is close to MCMC. With Δf0 = 0.2 µHz (near-full overlap) it is much broader and sometimes biased.
  - JS is higher and more widely spread than single-source for all parameters.
- App. 5 ablation: predicting the JKS amplitudes 𝒜^µ, whose posteriors are unimodal and near-Gaussian, is **worse** than predicting the physical (A, ι, φ0, ψ). It gives overly broad posteriors and also degrades sky localisation.

## Implementation details

- The JKS linear split lets one precomputed bank of phase-parameter bases serve as an unbounded amplitude/phase/noise augmentation.
- SNR-window rejection at training time.
- Order multi-source outputs to break label switching.
- φ0 ∈ [0, 2π] and ψ ∈ [0, π] are treated as **plain bounded intervals**, standardised. There is **no circular or combined φ0 ± 2ψ treatment**: the NSF learns the wrapped, multimodal φ0/ψ posteriors directly.

## Caveats

- Calibration is checked only by 1-D p-p plots at 1 mHz and JS vs MCMC on 100 injections. No SBC for the two-source or wide-band models.
- Narrow-band only (4 µHz). Training is very expensive.
- Fixed, known source count. Stationary noise, no gaps.
- The "unfavorable" two-source case fails.
