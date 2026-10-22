# Optimal, fast, and robust inference of reionization-era cosmology with the 21cmPIE-INN

Schosser et al., 2024, SciPost Phys. Core 8, 037 · [arXiv:2401.04174](https://arxiv.org/abs/2401.04174)

**Evidence: B.** Peer-reviewed (SciPost Physics Core); 13 citations in 2.8 yr (Semantic
Scholar, 2026-10-19).

## What they do

- BayesFlow-style NPE. A 3D CNN summary network (651k parameters) maps 140×140×2350 lightcones to a 6-D summary, which conditions a cINN (8 affine coupling blocks, 256-unit subnets, 33k parameters).
- The 6 parameters are Ωm, m_WDM, T_vir, ζ, L_X and E0, each normalized to [0,1].
- The summary is "physics-inspired": stage 1 trains it to regress the parameters by MSE, and joint training then lets it drift.

## What they find

- Simulation budget: 5000 lightcones (3600/400/1000), 900 GB in total.
- Table 2 (sim / mock), per parameter (Ωm, m_WDM, T_vir, ζ, L_X, E0):
  - R²: 0.981/0.990, 0.621/0.663, 0.764/0.716, 0.969/0.969, 0.987/0.973, 0.803/0.045
  - Err_cal: 0.007/0.045 for Ωm, up to 0.086 for T_vir (mock)
- With SKA thermal noise (21cmSense, optimistic foregrounds), E0 is lost entirely (R² 0.045). SBC shows a U shape for E0. The authors label this under-confidence, but a ∪-shaped rank histogram normally means over-confidence; their wording is inconsistent with the standard convention.
- Err_cal for m_WDM (0.011) looks good, yet the paper says the metric misses offsetting over- and under-estimation across the range. Integrated calibration metrics can hide local miscalibration.
- Ablation: without stage-1 regression pretraining, joint training "did not converge". The authors blame the 20× parameter imbalance between CNN and cINN.
- Stage 3 (joint training) lowers the cINN loss on the mock data to the noiseless level. The summary MSE to the true parameters rises, meaning the optimal summary departs from the parameter regression.

## Implementation details

- 3-stage schedule:
  - (1) Regress the summary by MSE: lr 4e-4, halved at epochs 15 and 20, 32 epochs.
  - (2) Freeze the summary and train the cINN: 150 epochs at constant lr 4e-4, which takes about 1 min per epoch.
  - (3) Joint training: lr ×0.3 every 5 epochs until epoch 20; 32 epochs (sim) / 68 (mock).
- Batch size 8; total training about 74 GPU-hours on a 1080 Ti.
- Data and architecture choices:
  - Filter out unphysical simulations (τ_reio within 5σ of Planck, x_HI(z=5) < 0.1) before training.
  - Asymmetric first kernel (3×3×102) with stride along the line of sight.
  - Max-pool in the spatial directions only.
  - Global average pooling before the dense layers.
- Validation: 100k posterior samples per observation, which takes seconds; SBC with 10k samples and 15 bins against a 99% binomial band; a latent-Gaussianity check.

## Caveats

- No MCMC ground truth, so "optimal" and "robust" are self-consistency claims (SBC, recovery), not tested against a reference posterior.
- Only the optimistic foreground model.
- Separate networks are trained for sim and mock data.
