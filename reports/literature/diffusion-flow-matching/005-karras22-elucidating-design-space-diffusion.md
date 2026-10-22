# Elucidating the Design Space of Diffusion-Based Generative Models

Karras et al., 2022, NeurIPS 2022 · [arXiv:2206.00364](https://arxiv.org/abs/2206.00364)

**Evidence: A.** Peer-reviewed (Neural Information Processing Systems); 4028 citations in 4.4
yr (Semantic Scholar, 2026-10-19).

## What they do

- Pulls apart choices that earlier papers bundled together: noise schedule σ(t), scaling s(t), sampler, preconditioning, training σ density p_train, and loss weight λ(σ).
- Denoiser D_θ(x;σ) = c_skip(σ)x + c_out(σ)F_θ(c_in(σ)x; c_noise(σ)). Loss: E[λ c_out² ‖F_θ − (y − c_skip(y+n))/c_out‖²].
- The scalings are derived from first principles (App. B.6):
  - c_in = 1/√(σ²+σ_d²): unit-variance network input.
  - c_out chosen so the effective target has unit variance.
  - c_skip = σ_d²/(σ²+σ_d²): the choice that minimises c_out, so network errors are amplified as little as possible.
  - This gives c_out = σσ_d/√(σ²+σ_d²).
  - λ = 1/c_out², so the initial loss is equal at every σ.
  - c_noise = ¼ ln σ, chosen empirically.
- Interpretation: the target slides from noise-prediction at small σ to x-prediction at large σ. ε-prediction would amplify errors by σ at high noise.

## What they find

- Table 2, CIFAR-10 FID with the deterministic sampler:
  | Config | Cond VP | Cond VE | Uncond VP | Uncond VE |
  |---|---|---|---|---|
  | A baseline | 2.48 | 3.11 | 3.01 | 3.77 |
  | B hyperparameters | 2.18 | 2.48 | 2.51 | 2.94 |
  | C capacity | 2.08 | 2.52 | 2.31 | 2.83 |
  | D preconditioning | 2.09 | 2.64 | 2.29 | 3.10 |
  | E loss (p_train, λ) | 1.88 | 1.86 | 2.05 | 1.99 |
  | F augmentation | 1.79 | 1.79 | 1.97 | 1.98 |
  - FFHQ-64 VE: 25.95 (A) → 3.39 (D) → 2.81 (E).
  - AFHQv2-64 VE: 18.52 (A) → 3.81 (D) → 2.28 (E).
  - Preconditioning on its own changes little on CIFAR-10. It is what made the network robust enough for the loss redesign in E to pay off, and it is decisive for VE at 64×64.
- Sampler (Heun 2nd order, σ(t) = t, s(t) = 1):
  - Needs 7.3× fewer NFE than the original VP sampler, 300× fewer than VE, and 3.2× fewer than DDIM.
  - Final models use 35 NFE at 32×32 and 79 NFE at 64×64.
  - Adaptive RK45 was not worth its cost.
- ImageNet-64: 2.07 → 1.55 from the sampler alone; retraining with config E reaches 1.36.
- Stochastic sampling helps older models, but with the improved training it *hurts* on CIFAR-10 and still helps on ImageNet-64.

## Implementation details

- Training density: ln σ ~ N(P_mean = −1.2, P_std = 1.2²); σ_data = 0.5; σ ∈ [0.002, 80].
- The reason for the log-normal density: the per-σ loss after training shows that only intermediate σ can be improved. Very small σ is "both difficult and irrelevant", and very large σ is trivially solved.
- Time steps: σ_i = (σ_max^{1/ρ} + i/(N−1)(σ_min^{1/ρ} − σ_max^{1/ρ}))^ρ with ρ = 7. ρ = 3 would equalise truncation error, but 5–10 works better because errors near σ_min matter most.
- Stochastic sampler: add noise via S_churn inside [S_tmin, S_tmax], with S_noise slightly above 1. These are grid-searched per model.
  - Too much churn erodes detail. The paper attributes this to a non-conservative denoiser that over-removes noise (regression to the mean).
- Lengthen the ODE time grid, not the network: deterministic Heun with ρ-spaced steps is the default.

## Caveats

- Images only. Hyperparameters (P_mean, P_std, ρ, the churn settings) were tuned per dataset.
- The σ_d preconditioning assumes one global data scale, which a conditional problem does not have.
- Total project compute is about 250 MWh.
