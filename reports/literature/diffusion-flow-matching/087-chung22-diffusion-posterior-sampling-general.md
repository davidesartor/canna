# Diffusion Posterior Sampling for General Noisy Inverse Problems

Chung et al., 2022, ICLR 2023 · [arXiv:2209.14687](https://arxiv.org/abs/2209.14687)

**Evidence: A.** Peer-reviewed (International Conference on Learning Representations); 2039
citations in 4.1 yr (Semantic Scholar, 2026-10-19).

## What they do

- Approximate the intractable p(y | x_t) by p(y | x̂0(x_t)), using the Tweedie posterior mean.
- The approximation error is a Jensen gap, bounded in closed form for Gaussian noise.
- Add the gradient −ζ_i ∇_{x_t}||y − A(x̂0)||² to each ancestral step, with no projection. It handles Gaussian and Poisson noise and nonlinear A (phase retrieval, non-uniform blur).

## What they find

- FFHQ-256, FID (LPIPS):
  - SR×4: DPS 39.35 (0.214), DDRM 62.15, MCG 87.64.
  - Box inpainting: 33.12.
  - Gaussian deblurring: 44.05.
- ImageNet: SR×4 50.66.
- Phase retrieval works, but not always: non-uniqueness makes it initialisation-dependent.

## Implementation details

- The step size is not the principled ρ = 1/σ² but ζ_i = ζ′/||y − A(x̂0(x_i))||, with ζ′ tuned per task (App. C).
- Differentiate through the denoiser (one backprop per step).

## Caveats

- The Jensen-gap approximation plus the heuristic, residual-normalised step size mean the sampler does not target the exact posterior.
- Evaluation is FID and LPIPS only, with no coverage or calibration.
- The paper never claims calibration, and its tuned ζ′ is a likelihood-weight knob, so DPS samples are not calibrated posteriors.
