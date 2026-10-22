# Multifield Cosmology with Artificial Intelligence

Villaescusa-Navarro et al., 2021, arXiv preprint in PNAS format · [arXiv:2109.09747](https://arxiv.org/abs/2109.09747)

**Evidence: B.** Not peer-reviewed; 28 citations in 5.1 yr (Semantic Scholar, 2026-10-19).

## What they do

- A CNN maps 256×256 projected maps (25 h⁻¹ Mpc on a side, 5 h⁻¹ Mpc thick, z=0) straight to the marginal posterior mean and variance of 6 parameters (Ω_m, σ8 and 4 feedback amplitudes).
- It is trained as a "moment network" on 1,000 IllustrisTNG LH simulations, marginalising baryonic feedback at the field level.
- It is run on 13 single fields, on an 11-channel "multifield" (MF) and on gravity-only N-body maps.

## What they find

- Gas temperature T: ⟨δΩ_m/Ω_m⟩ ≈ 3.5% and ⟨δσ8/σ8⟩ ≈ 5.3%. A_AGN1 is not learned at all (the network predicts the prior mean).
- Ω_m precision by field:
  - M_dm 2.9%; N-body 3% (N-body also gives the best σ8, ~2%); Z 8.8%; Mg/Fe 11%.
  - M_* is worst, attributed to sparsity (most pixels are zero).
- MF vs single fields:
  - Ω_m: 35% better than M_tot and 15% better than M_dm. It also beats N-body on Ω_m, attributed to cross-correlations beating cosmic variance and to Ω_b information in the gas.
  - A_SN1 and A_SN2: ~40% better than the best single field, Z.
  - σ8: no gain over M_dm.
- Main negative result: a T-trained network tested on SIMBA maps fails badly, with both the mean and σ wrong. The companion paper 2109.10360 finds M_tot is robust across suites.

## Implementation details

- Moment-network loss (Jeffrey & Wandelt): Σ_i log Σ_batch (θ−μ)² + Σ_i log Σ_batch ((θ−μ)² − σ²)².
  - The **log of each per-parameter batch sum** stops unlearnable parameters (AGN) from dominating gradient noise and swamping the learnable ones.
  - The authors say that without the log, training on the easy parameters stalls.
- Architecture and training:
  - 6 conv blocks, each with BatchNorm and LeakyReLU, halving the spatial size and doubling the channels, then 2 FC layers.
  - AdamW with a cyclic LR.
  - Optuna with 50 trials × 500 epochs, batch 128, tuning width, LR, weight decay and dropout.
- Data handling:
  - Split by simulation (900/50/50) so the 15 maps from one sim never leak across splits.
  - Augmentation with the 8 rotations and flips.

## Caveats

- Marginal mean and σ only: no joint posterior and no coverage test. Calibration is shown only as scatter plots of error bars.
- Noiseless, not directly observable fields; tiny volume.
- Cross-simulator failure on T shows that a high-information field-level network can latch onto simulator-specific features.
