# A Fourier Space Perspective on Diffusion Models

Falck, Pandeva, Zahirnia, Lawrence, Turner, Meeds, Zazo, Karmalkar; Microsoft Research; arXiv v1 May 2025, ICML-formatted · [arXiv:2505.11278](https://arxiv.org/abs/2505.11278)

**Evidence: B.** Not peer-reviewed; 58 citations in 1.4 yr (Semantic Scholar, 2026-10-19).

## What they do

In Fourier space DDPM gives per-frequency SNR s_t(i) = ᾱ_t C_i/(1−ᾱ_t), C_i = Var[(y₀)_i], which follows a power law. So high frequencies are noised earlier (lower SNR at every t) and faster (larger change in SNR per step). Prop 1 (counterexample): for a two-component mixture ½N(−1,δ²)+½N(1,δ²) with added noise N(0,4), q(x_{t−1}|x_t) stays a constant TV distance from any Gaussian. So when the per-step noise is large relative to the signal variance, the Gaussian reverse kernel is wrong. Measured KL to the nearest Gaussian at t=1 on CIFAR-10: high frequency 1.8e−2 vs low frequency 3.1e−4 under DDPM; under EqualSNR 1.3e−4 vs 1.0e−4. **EqualSNR (Def 2):** y_t = √ᾱ_t y₀ + √(1−ᾱ_t) ε_Σ with Σ_ii = c·C_i. This is iff for equal SNR across coordinates; with Σ = c·Cov(y₀) it holds in every basis. In practice this is diffusion in *whitened* coordinates. Loss ‖C^{−1/2}(y₀ − ŷ₀)‖² (x₀-prediction, Prop 2: an ELBO). The U-Net stays in pixel space: ŷ₀ = F∘f_θ(F⁻¹ y_t, t). **Calibration (A.3):** match the frequency-averaged SNR, ᾱ^eq/(1−ᾱ^eq) = ᾱ^ddpm·mean(C)/(1−ᾱ^ddpm). FlippedSNR (high frequencies generated first) "did not succeed in numerous experiments".

## What they find

- **Clean-FID, Table 1, at T = 50/100/200/1000 sampling steps:**
  - CIFAR-10: DDPM 18.63/18.01/17.68/17.70; EqualSNR-calibrated 16.00/15.91/15.76/15.73; DDPM-calibrated 16.64/14.69/14.07/13.85.
  - CelebA-64: DDPM 10.10/8.72/8.30/8.62; EqualSNR-calibrated 9.45/8.79/8.62/8.56; DDPM-calibrated 6.59 at T=1000.
  - LSUN Church-128: DDPM 29.36/25.36/24.03/23.22; EqualSNR-calibrated 19.42/19.75/19.90/19.80; DDPM-calibrated 40.31/…/20.09.
- **PyTorch-FID, Table 4:** CIFAR-10 at T=1000: DDPM 4.81, EqualSNR-calibrated 4.27, DDPM-calibrated 3.71. LSUN Church: 16.63 vs 13.61 vs 15.3.
- **Takeaway on FID:** EqualSNR saturates with fewer steps and is best at small T and high resolution, but is not uniformly best.
- **High-frequency discriminability (Table 2):** a logistic classifier on 2 spectral-fit parameters from the top 5/15/25% band reaches accuracy 0.62–0.65 for DDPM, with 99–100% of tests rejected at p=0.05. For EqualSNR accuracy is 0.52 and 10–16% of tests are rejected. Low-frequency bands show no difference between the two.
- **Dots (32², 46–50 white pixels):** DDPM over-generates dots; EqualSNR matches the count.

## Implementation details

- Define the forward process by per-coordinate SNR rather than by mixing coefficients; this makes it resolution-agnostic, with no simple-diffusion-style shift needed.
- When comparing processes, calibrate the *average* SNR per t.
- Scale the noise by √C_i and weight the loss by C^{−1}.
- Keep the network in the native (pixel) domain; feeding Fourier coefficients directly to the U-Net was uncompetitive.

## Caveats

- Largest resolution is 128²; only image benchmarks.
- The FID ordering at T=1000 favours DDPM-calibrated on CIFAR/CelebA.
- The diagonal-C approximation is exact only for stationary data.
- EqualSNR loses the coarse-to-fine hierarchy, which matters for perceptual quality at scale (they cannot rule FlippedSNR out, but it failed).
