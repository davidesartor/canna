# Diffusion-based mass map reconstruction from weak lensing data

Boruah, Jacob & Jain, 2025, arXiv preprint · [arXiv:2502.04158](https://arxiv.org/abs/2502.04158)

**Evidence: B.** Peer-reviewed (Physical Review D); 8 citations in 1.7 yr (Semantic Scholar,
2026-10-19).

## What they do

- Train one unconditional DDPM on κTNG-like MassiveNuS convergence maps: 3 tomographic bins as channels, 256², 8,000 maps.
- Use it both as an emulator and, through DPS, as the prior for mass mapping. DPS approximates the likelihood with Tweedie x̂0(x_t): ∇ log p(d | x̂0(x_t)).
- Fix DPS bias by scaling the likelihood gradient with A(t) = sigmoid((t̄ − t)/σ_t), with t̄ = 500 and σ_t = 20 on a 1000-step clock. The likelihood is switched off for the noisy first half of sampling.

## What they find

- Emulator: power-spectrum bias of a few percent; first- and second-order scattering transforms match; the summary-statistic covariance matrix and its eigenvalues match the simulations.
- Standard DPS: early latents leave the prior's typical set (diagnosed with the scattering transform S1 of x_t at each step, fig. 4). This gives strongly biased power spectra and scattering transforms at t = 0. With the rescaling, residual biases are only a few percent.
- RMSE / ρ_c compared with KS smoothed at 1 arcmin:
  | Bin | DPS RMSE | KS RMSE | DPS ρ_c | KS ρ_c |
  |---|---|---|---|---|
  | 1 | 0.014 | 0.027 | 0.58 | 0.38 |
  | 2 | 0.020 | 0.027 | 0.62 | 0.55 |
  | 3 | 0.026 | 0.028 | 0.61 | 0.61 |
  About 50% better in the lowest-SNR bin.
- Cost: about 50 s per posterior sample and about 10 s per prior sample on a V100.

## Implementation details

- Preprocessing: inverse lognormal transform y = log(κ + κ_min), then min-max scaling to [0, 1] per bin. This gaussianises the target and helps training.
- Model: lucidrains denoising-diffusion-pytorch UNet, 35M parameters, 4 levels (64 → 512 channels), linear attention per level, sinusoidal time MLP. Sigmoid β schedule, T = 1000, 300k iterations.
- Diagnostic worth copying: track a summary statistic of the intermediate states x_t against the same statistic under the forward-noised prior p_t. Divergence early on predicts final-sample bias.
- The schedule is tuned empirically. There is no principled choice of t̄ or σ_t.

## Caveats

- The A(t) parameters are hand-tuned on the same mocks used for evaluation.
- Validation uses only summary statistics. No coverage.
- Only one patch is shown for the power-spectrum bias.
- The prior is at a single cosmology (fiducial MassiveNuS).
- The DPS posterior is not exact even with the rescaling.
