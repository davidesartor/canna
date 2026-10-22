# Probabilistic Mass Mapping with Neural Score Estimation

Remy et al., 2022, A&A 672, A51 · [arXiv:2201.05561](https://arxiv.org/abs/2201.05561)

**Evidence: B.** Peer-reviewed (Astronomy & Astrophysics); 38 citations in 4.8 yr (Semantic
Scholar, 2026-10-19).

## What they do

Model the posterior score as the sum of two parts:
- an analytic Gaussian likelihood score;
- a hybrid prior score = the analytic Gaussian (power-spectrum) prior score + a neural residual r_θ trained by denoising score matching on κTNG maps.

## What they find

- Gaussian-prior validation: the MAP and the posterior mean (1,500 samples) match the Wiener filter. Residuals are two orders of magnitude below the signal, and power-spectrum errors are ≈0.
- Mock COSMOS, RMSE / Pearson r:
  | Method | RMSE | r |
  |---|---|---|
  | Kaiser-Squires | 2.40e-2 | 0.57 |
  | Wiener filter | 2.31e-2 | 0.61 |
  | GLIMPSE | 2.84e-2 | 0.42 |
  | MCALens | 2.19e-2 | 0.67 |
  | DeepMass | 2.18e-2 | 0.68 |
  | DLPosterior mean | 2.16e-2 | 0.68 |
- Posterior-sample power spectra match κTNG; prior-only (masked) regions have high variance.
- Cluster detection frequency across posterior samples increases monotonically with input SNR.
- Cost: about 10 V100 GPU-minutes per independent sample.

## Implementation details

- Hybrid prior: the network learns only the non-Gaussian residual, with ∇log p_Gauss as an input. Trained with the AR-DAE residual loss ‖u + σ r_θ(x + σu, σ)‖², σ ~ |N(0, 0.2²)|.
- Network: U-net with ResNet blocks.
- Spectral normalisation keeps the score Lipschitz and well-behaved out of distribution.
- Adam, lr 1e-4 → 1e-7, batch 32, 40k steps.
- Masked pixels get noise variance 1e10 instead of infinity.
- The likelihood is "convolved" to variance σn² + σ² at each temperature σ², so likelihood and prior anneal together.
- Annealed HMC uses an MH correction computed from the score.
- The chains cannot go below σ ≈ 1e-3, so a probability-flow ODE from that temperature finishes the denoising.
- Chains are initialised from white noise, one chain per sample.
- Validation strategy: (i) an analytically tractable Gaussian limit (the Wiener filter) checks the sampler; (ii) an independently trained amortised posterior-mean estimator (DeepMass, same U-net) must match the sample mean.

## Caveats

- The convolved likelihood × convolved prior is not the convolution of the posterior. Justified empirically only.
- Slow: no amortisation.
- The prior is at a fixed cosmology, so the posterior is biased toward the fiducial cosmology on scales the data barely constrain.
- Validation is through posterior means and power spectra, not coverage.
