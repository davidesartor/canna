# Not just a phase: detecting nanohertz gravitational waves from phase alone

Grunthal et al., 2026, arXiv preprint (A&A style) · [arXiv:2610.06178](https://arxiv.org/abs/2610.06178)

**Evidence: C.** Not peer-reviewed; 0 citations in 0.0 yr (Semantic Scholar, 2026-10-19). Its
claims are the authors' alone: a hypothesis, not a result.

## What they do

- Write the Fourier residuals as z = 𝒜 e^{iφ} and marginalise the amplitudes 𝒜 to get a phase-only likelihood. Each phase is marginally uniform; the GWB shows up only as a copula of correlations between phase differences Δφ.
- For N = 2 there is a closed form, p(Δφ) ∝ [1 − r cosΔφ · arccos(r cosΔφ)]/√(1 − r² cos²Δφ), which gives cosine "stripes".
- For N > 2 the positive-orthant integral is intractable, so it is learned with a conditional NCSF.
- Under H0 the null is exactly uniform, so the null distribution of Bayes factors is obtained by evaluating on U(−π, π)^N draws, without data resampling.

## What they find

- Setup: 30 pulsars, 12 bins, white noise 50 ns, injection log10 A = −13.5 (the text says −14.0 in one place), γ = 13/3.
- The phase likelihood recovers (A, γ). Its p-value of 0.00047 (3.3σ) comes from 1e5 null draws.
- Compared with the optimal statistic: slightly wider 2-D contours, but better A–γ separation. The OS amplitude is tighter at fixed γ, which shows the OS leaks amplitude information.
- Big realisation-to-realisation scatter; some draws give non-detections.
- Factorised in frequency (a 30-D flow per bin): P-P plot fine.
- Joint 240-D flow (30 pulsars × 8 bins; best of 12 layers with (32,32) MLPs): P-P plot fails. A mean shift in A and a γ distribution that is too narrow bias the averaged posterior to lower A and steeper γ.

## Implementation details

- Circular data needs a circular spline flow, with continuity at ±π up to first derivative.
- Parameter points are drawn on a Sobol sequence (avoids gridding artefacts, gives better coverage) with N_sample = 15k–30k phase draws per point.
- Three training rounds with decreasing lr: 1e-3, 3e-4, 1e-4. N_vp goes 500 → 1,000 → 1,000. Each round stops when the relative change in validation NLL stays below a threshold (5e-3 → 1e-4 → 5e-5) for 5 epochs.
- Model: 15 layers × 8-bin splines × [64, 64] MLPs. AdamW.
- Validation: P-P plots of (A, γ) from 1,000 evaluations, plus posteriors averaged over 100–2,000 noise realisations.

## Caveats

- Toy PTA: regular sampling, known white noise, no inter-bin correlations.
- The authors stress that the flow setup is fragile: adding or removing 5 pulsars can require architecture changes.
- Learning only A was easy; conditioning on both A and γ "proved to be hard".
- The high-dimensional joint flow is miscalibrated.
