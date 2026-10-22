# Analyzing and Improving the Training Dynamics of Diffusion Models

Karras et al., 2024, CVPR 2024 · [arXiv:2312.02696](https://arxiv.org/abs/2312.02696)

**Evidence: A.** Peer-reviewed (Computer Vision and Pattern Recognition); 484 citations in 2.9
yr (Semantic Scholar, 2026-10-19).

## What they do

- In ADM/EDM, activation and weight magnitudes drift without bound during training. This silently changes the effective learning rate per layer and the relative weight of noise levels.
- Fix: keep every layer's activation, weight and update magnitudes at 1 in expectation, by construction.
- Two independent contributions:
  - Learned per-σ loss weighting, a continuous version of Kendall et al.'s homoscedastic multi-task uncertainty (App. B.2).
  - Post-hoc EMA reconstruction (Sec. 3).

## What they find

- Table 1, ImageNet-512 latent, FID without guidance:
  | Config | Change | FID |
  |---|---|---|
  | A | EDM baseline | 8.00 |
  | B | + minor improvements, including u(σ) weighting and retuned p_train | 7.24 |
  | C | + streamlining (no biases, cosine attention) | 6.96 |
  | D | + magnitude-preserving learned layers (weight norm) | 3.75 |
  | E | + forced weight norm, inverse-sqrt lr | 3.02 |
  | F | − group norms | 2.71 |
  | G | + magnitude-preserving fixed-function layers | 2.56 |
- Table 2, ImageNet-512, FID no-CFG / with CFG, all at 63 NFE with deterministic sampling:
  | Model | Params | No CFG | With CFG |
  |---|---|---|---|
  | EDM2-XS | 125M | 3.53 | 2.91 |
  | EDM2-S (= config G) | 280M | 2.56 | 2.23 |
  | EDM2-L | 777M | 2.06 | 1.88 |
  | EDM2-XXL | 1.5B | 1.91 | 1.81 |
  - The previous records were 2.99 without guidance (VDM++) and 2.41 with it.
- ImageNet-64, deterministic sampling: EDM2-S 1.58, EDM2-L 1.33. Compare EDM at 2.22.

## Implementation details

- Uncertainty weighting, Eq. 21: L = E_σ[λ(σ)L(D;σ)/e^{u(σ)} + u(σ)].
  - u is a one-layer MLP on Fourier features of c_noise(σ), trained jointly with the network and thrown away afterwards.
  - Its optimum is u = ln L(D;σ) + ln λ(σ), so each σ's gradient is divided by its current loss. Contributions are equalised across σ *and over training time* (Eq. 28).
  - Motivation: EDM's λ equalises the loss only at initialisation.
- Config B noise density is ln σ ~ N(−0.4, 1.0²) for latents; lr 2e-4, batch 2048.
- Weight handling:
  - Forced weight normalisation: re-normalise each weight row to unit norm *before every step*, and also normalise on use.
  - This projects gradients onto the tangent plane of the sphere and keeps Adam's statistics clean.
  - Unit-Gaussian init.
  - lr α_ref/√max(t/t_ref, 1).
- Activations:
  - MP-SiLU divides by 0.596; Fourier features are scaled by √2.
  - Residual adds are fixed weighted sums (t = 0.3 in blocks, 0.5 in the embedding), divided by the expected std.
  - A learned zero-init gain at the output and on the conditioning in each block; dropout is no longer needed in small models.
- Cosine attention (normalise q, k) allows fp16 throughout.
- Post-hoc EMA:
  - The averaging profile is a power function θ̂_γ(t) ∝ ∫τ^γθ(τ)dτ, which scales with training length and gives zero weight to the init. Update rule: β_γ(t) = (1 − 1/t)^{γ+1}.
  - The width is quoted as σ_rel, e.g. 10% ⇔ γ ≈ 6.94.
  - Keep two averages (σ_rel = 0.05 and 0.10) and save a snapshot every 4096 steps. Any profile is then a least-squares combination of the snapshots.
- Post-hoc EMA findings:
  - The optimal EMA length depends strongly on guidance strength.
  - The optimal length shrinks with higher lr (σ_rel ∝ 1/(α_ref² t_ref)), with larger models, and with simpler data.
  - In config B, per-tensor EMA lengths disagree; in config G they agree.

## Caveats

- Images and U-Net only. Whether the magnitude-preserving design transfers to DiT/MMDiT is stated as open (Sec. 5).
- The u(σ) gain is bundled into config B with other tuning, so its own FID effect is not isolated.
- Post-hoc EMA observations are called "anecdotal" by the authors.
