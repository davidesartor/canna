# Listening to the Noise: Blind Denoising with Gibbs Diffusion

Heurtel-Depeiges et al., 2024, ICML 2024 · [arXiv:2402.19455](https://arxiv.org/abs/2402.19455)

**Evidence: B.** Peer-reviewed (International Conference on Machine Learning); 7 citations in
2.6 yr (Semantic Scholar, 2026-10-19).

## What they do

- The data model is y = x + ε, with ε ~ N(0, Σ_ϕ).
- Train a **ϕ-conditional diffusion model whose forward noise has the covariance Σ_φ itself** (coloured noise, not white). The observation y, rescaled, is then exactly a diffusion state z_{t*}. Running the reverse SDE from t* samples p(x | y, ϕ).
- Gibbs-alternate this with HMC on p(ϕ | ε = y − x), which uses the analytic Gaussian likelihood.
- Cosmology use: CMB = "noise" with (σ, H0, ω_b); dust = signal.

## What they find

- Blind denoising on ImageNet and CBSD68: the posterior mean beats non-blind BM3D and DnCNN in PSNR. Example: 32.2 vs 31.0 dB at σ = 0.06 with pink noise.
- Single posterior samples score about 3 dB below the posterior mean (expected).
- Convergence: 40 Gibbs iterations × 16 chains, about 20 to converge. R̂ ≲ 1.1 and ESS ≈ M.
- **SBC finds bias** in the spectral index φ (images) and in the CMB amplitude σ, while H0 and ω_b are near-uniform. The bias is attributed to diffusion-model error, quantified by Theorem 2.3: the KL between the true and the Gibbs-stationary marginal equals E log E[q/p].
- Appendix E: DPS and Π-GDM fail on the CMB problem. Σ_ϕ has condition number about 6e5, and Σ^{-1/2} has eigenvalues up to 160. Guidance overflows; damped guidance ruins the large scales.

## Implementation details

- Loss renormalised by −√(1 − ᾱ) Σ_ϕ (effectively ε-prediction in whitened units) **to survive a condition number of about 1e4**.
- The parameters are embedded like time: linear + activation, then a per-ResBlock MLP, added to the input.
- Circular padding. No attention on the small dataset (991 maps).
- **A dedicated low-noise model (β_max = 2 instead of 20)**, used only in the posterior regime, improves summary statistics and calibration. Fourier features of the input help at low noise.
- **Training long past the loss plateau markedly improved SBC calibration**: 100k epochs, 96 h on an A100.
- Initialisation: a moment network for E[ϕ | y] initialises the chains, projected into the prior box (chains outside the prior get stuck).
- HMC details:
  - random leapfrog count U{5..15};
  - **elastic reflection at the prior boundaries**;
  - dual-averaging step size (target acceptance 0.65);
  - mass matrix estimated from warm-up.
- A differentiable neural emulator of ϕ → Σ_ϕ replaces CAMB.
- Checks for dataset copying with a periodic L2 distance.

## Caveats

- Gaussian noise only.
- Only 2–3 parameters.
- Residual SBC bias.
- Expensive per observation (MCMC × diffusion).
