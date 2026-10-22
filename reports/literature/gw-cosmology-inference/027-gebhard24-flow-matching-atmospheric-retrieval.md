# Flow Matching for Atmospheric Retrieval of Exoplanets: Where Reliability meets Adaptive Noise Levels

Gebhard, Wildberger et al., 2024, A&A (accepted Oct 2024) · [arXiv:2410.21477](https://arxiv.org/abs/2410.21477)

**Evidence: B.** Peer-reviewed (Astronomy & Astrophysics); 11 citations in 2.0 yr (Semantic
Scholar, 2026-10-19).

## What they do

use FMPE (a CNF trained by flow matching) for exoplanet atmospheric retrieval.
- The models are conditioned on the assumed noise level σ as well as the spectrum, so one model serves any error bar.
- Every result is importance-sampled against the exact Gaussian likelihood. IS gives three things: correction, the ε diagnostic, and the evidence.

## What they find

- **Setup:** 16 parameters, uniform priors, 379 spectral bins (R = 400, petitRADTRANS).
  - Training set: 2²⁵ ≈ 33.6M simulated spectra, which cost ~12,500 CPU-h.
  - Both models have ~310M parameters.
- **Benchmark spectrum (FMPE-IS vs NPE-IS):**
  - Noise-free: ε = 19.1% vs 11.6%.
  - Noisy: ε = 10.6% vs 5.0%.
  - log Z agrees with nautilus to about 0.001–0.06. MultiNest is off by ~5 in log Z and overconfident.
- **Misspecified σ:** data with σ = 0.1 analysed as σ = 0.5 gives ε = 22.6%. Data with σ = 0.5 analysed as σ = 0.1 gives ε = 1.5e-5, so ε flags the misspecification.
- **Calibration:** P–P plots on 500 prior draws are well calibrated for both FMPE and NPE.
- **FMPE vs NPE on 500 local tests:**
  - Their ε values are correlated (ρ = 0.70), consistent with the embedding being the shared bottleneck.
  - FMPE has higher ε on average and lower JSD between the pre-IS and post-IS posteriors for 15 of 16 parameters.
  - log Z from FMPE-IS and NPE-IS agree within 1σ in more than half the cases and within 3σ in 98%.
- **Inference cost for 65,536 samples:** FMPE takes 16 s to sample and 258 s to sample with log-density. NPE takes 1.16 s for either. FMPE is therefore ~10× slower to sample and ~200× slower for densities.

## Implementation details

- **Noise-level conditioning:**
  - For each training example, draw σ ~ U(0.05, 0.5), add n ~ N(0, σ²I) on the fly, and feed (x + n, σ).
  - Noise is resampled every time from a fixed bank of noise-free spectra, which is effectively data augmentation.
- **Mass-covering argument:** both NPE and FMPE are mass-covering, which is what makes IS valid.
- **IS practice:**
  - Use 2²⁰ proposal samples for headline results and 1e5 per retrieval in the large-scale tests.
  - Do the weights in log space.
  - Treat ε ≳ 1% as "good". ε is very sensitive to deviations in a single dimension, so low ε does not necessarily mean a bad proposal.
- **Gaussian probability path:** σ_min path p_t = N(θ₁ t, (1 − (1 − σ_min)t)²).
- **Independent reparametrisation of ordered parameters:** the T1 < T2 < T3 constraint becomes independent U[0, 1] fractions (T3 ← T_c(1 − u3), T2 ← T3(1 − u2), ...). This gives independent priors with ordering by construction.
- **Two test sets:** draws from the prior (for calibration) and a Gaussian neighbourhood of a benchmark (for ε statistics).
- **Proposed extensions:** prior conditioning (citing DINGO-BNS 2407.09602) for parameters known from other sources, and conditioning on GP-kernel noise descriptions.

## Caveats

- A low-dimensional, smooth, single-source problem with modest SNR. It does not stress the narrow-posterior regime.
- Training cost and parameter count are large (310M).
- The FMPE density evaluation needed for IS is expensive: ~200× slower than NPE.
- The training-set-size ablation sits in the appendix I could not read.
