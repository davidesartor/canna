# Diffusion-HMC: Parameter Inference with Diffusion-model-driven Hamiltonian Monte Carlo

Mudur et al., 2024/25, ApJ 978, 64 · [arXiv:2405.05255](https://arxiv.org/abs/2405.05255)

**Evidence: B.** Peer-reviewed (Astrophysical Journal); 8 citations in 2.4 yr (Semantic
Scholar, 2026-10-19).

## What they do

- Train a conditional DDPM emulator p_φ(x0 | θ) of CAMELS dark-matter fields on (Ω_m, σ8).
- Use its **truncated variational lower bound**, −Σ_{t<T_MAX} L_t, as a differentiable surrogate for −log p(x0 | θ).
- Run HMC over θ with that surrogate. This is a learned-likelihood (NLE-like) approach, not NPE.

## What they find

- Per-timestep likelihood contours (the 1σ contour of −2Δln L_t on a 50² grid) all centre on the truth, and **widen as t grows**: the small-noise terms carry most of the parameter information.
- Truncating at T_MAX = 20 of 1000 steps shifts the means by only −0.26% (Ω_m) and −1.27% (σ8).
- Cost: 500 HMC samples take about 32 min per field.
- Accuracy on 10 validation fields: mean bias −0.006 on Ω_m (2%) and 0.01 on σ8 (1.26%).
- **Under-dispersed**: mean |z| = 2.75 for Ω_m and 1.14 for σ8. The authors blame the truncation and the single-seed VLB estimate.
- Much tighter than a P(k) + mean NPE baseline (a MAF).
- More robust than a direct regression network (InfNet) to added white noise of σ = 0.01–0.02: InfNet shifts by 6–23%.
- Emulator quality:
  - χ²_red of P(k) is 2.29 ± 0.49, against a leave-one-out baseline of 1.70 ± 0.36 between true fields.
  - Cosmic-variance set: mean χ² 36.6 (generated) vs 33.4 (true), with 35 degrees of freedom.
  - The P(k) spread is slightly under-predicted at large scales (mild mode collapse).

## Implementation details

- U-Net with circular convolutions for the periodic box. The parameters are normalised to [0, 1] and passed through an MLP into the time-embedding space; each ResNet block gets its own conditioning MLP.
- Huber loss on ε-prediction. Random flips and rotations.
- **Progressive resolution**: pretrain at 64², then initialise the 256² model from it, which converges faster.
- **Checkpoint selection by a physics statistic** (χ²_red of P(k) on generated samples), because the denoising loss does not track sample quality.
- HMC details:
  - Hand-set diagonal inverse mass matrix [1, 5] and step 5e-4, dropped to 1e-4 near the prior edge.
  - Single noise seed per evaluation, a fresh seed each step.
  - VLB gradients accumulated over batches of 10 timesteps.
  - R̂ = 0.997, burn-in 100.
- Appendix A.3: dropping the first two timesteps adds robustness to known noise levels, playing the role of a scale cut.

## Caveats

- Only 2 parameters; tiny 25 Mpc/h boxes; 10 test fields.
- Calibration is poor (under-dispersed) and there is no SBC/TARP.
- The VLB is a bound, not the likelihood. Stochastic single-seed gradients inside HMC are formally invalid: no Metropolis correction on a noisy target.
- Slow: 32 min per field.
