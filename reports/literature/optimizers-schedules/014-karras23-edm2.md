# Analyzing and Improving the Training Dynamics of Diffusion Models, EDM2 (Karras et al., 2023)

[arXiv:2312.02696](https://arxiv.org/abs/2312.02696), CVPR 2024.

**Evidence: A.** Peer-reviewed (Computer Vision and Pattern Recognition); 484 citations in 2.9
yr (Semantic Scholar, 2026-10-19).

## What they do

- **Magnitude-preserving layers**, so activations keep unit scale through the network.
- **Forced weight normalisation**: weights are renormalised every step, with no weight decay,
  and the lr decays as inverse square root.
- **Learned per-noise-level loss weighting**: a log-variance u(σ), giving the loss
  e^{−u}·L + u.
- **Post-hoc power-function EMA.**
  - The averaged weights at step t weight each earlier step τ by τ^γ.
  - Its width is σ_rel = √((γ+1) / ((γ+2)²(γ+3))):

    | γ | σ_rel |
    |---|---|
    | 0 (uniform average) | 0.29 |
    | 1 | 0.24 |
    | ≈ 7 | 0.10 |

  - Two EMA profiles are stored during training (σ_rel ≈ 0.05 and 0.10). Any σ_rel can then
    be reconstructed by least squares after training.

## What they find

- ImageNet-512 FID improves from 2.41 to 1.81.
- **The best EMA length matters a lot, and its optimum is sharp.**
  - FID is best at σ_rel of about 0.07–0.13.
  - The best value depends on the model size and on the lr.
  - Post-hoc reconstruction finds it without retraining.
- **Weight growth is an uncontrolled lr decay.** Without forced normalisation, weight
  magnitudes grow during diffusion training. Behind normalisation layers, the growth shrinks
  the effective lr unevenly across layers. Forcing unit-norm weights removes it.
- The diffusion literature evaluates EMAs of the training iterate, not schedule-free
  averages.

## Limits

- Image diffusion with a decaying lr. The EMA is a passive average and does not feed back into
  the gradient point.
