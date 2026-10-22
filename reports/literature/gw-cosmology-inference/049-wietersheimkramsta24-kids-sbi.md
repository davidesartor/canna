# KiDS-SBI: Simulation-Based Inference Analysis of KiDS-1000 Cosmic Shear

von Wietersheim-Kramsta et al., 2024, A&A · [arXiv:2404.15402](https://arxiv.org/abs/2404.15402)

**Evidence: B.** Peer-reviewed (Astronomy & Astrophysics); 39 citations in 2.5 yr (Semantic
Scholar, 2026-10-19).

## What they do

A full SBI of KiDS-1000 cosmic-shear pseudo-C_ℓ (120-D: 8 ℓ-bins × 15 tomographic pairs). The forward model uses GLASS log-normal shells (19 shells, Nside 1024), Levin non-Limber projection, NLA intrinsic alignment, variable depth tied to a single σ_rms map, and PSF/additive/multiplicative shear biases drawn per realisation. The 120-D vector is score-compressed (Gaussian likelihood, 1600-sim fiducial covariance, five-point-stencil derivatives = MOPED) to 7-D. The likelihood is learnt with pyDELFI using an ensemble of 6 MAFs (3–8 MADEs, 2×50 hidden) and sampled with nautilus. Nuisances such as δ_z and the 20 shear-bias parameters are implicitly marginalised: they are drawn from their priors in each sim and are not in the conditioning or summaries.

## What they find

- Headline: S8 = 0.731 ± 0.033, which is 2.9σ from Planck.
- Sim cost: one sim takes ~20 min on one core, vs ~280 min for the FLASK/TreeCorr equivalent.
- Simulation budget:
  - 2000 Latin-hypercube points to start, then active learning.
  - The loss stopped improving after ~5000 sims; 18,000 were run "to densely sample the prior".
  - The precursor paper (Lin et al. 2023) found active learning halves the sims needed vs a Latin hypercube alone.
- Likelihood shape ablation, on the same 18k sims and the same noisy mock:
  - A single-Gaussian MDN ensemble (whose covariance varies with θ) matches the MAF posterior almost exactly.
  - A fixed-covariance Gaussian likelihood is ~10–11% narrower in S8 and more biased (0.744 vs SBI 0.762, truth 0.756).
  - The cause is the S8-dependence of cosmic variance (App. F). So non-Gaussianity is not the issue; the cosmology dependence of the noise is.
- Systematics ablation (anisotropic model vs standard isotropic model, each with its own 18k sims):
  - The signal shift is only ~1% (~0.2–0.3σ).
  - Yet S8 on the same noisy mock moves 0.7σ, and on real data 0.9σ (0.731 vs 0.772). Systematics change the noise model, which re-weights the data.
  - Bayes factor |ln B| = 1.11: the data show no preference between the two models.
- Coverage: TARP on 500 new sims (S8 and Ω_m only), bootstrapped 100×, shows slight under-confidence, within 1σ.
- Re-compressing at the first-pass MAP changed the posterior only marginally.
- Posterior scatter across noise realisations: S8 mean ±0.5σ, A_IA ±0.75σ (App. H).
- Sampler check: multinest gives ~15% narrower S8 than nautilus on the same likelihood (App. G).

## Implementation details

- The 1600 fiducial sims serve three uses: the compression covariance, the χ² goodness-of-fit (GoF) mean and covariance, and the noise model.
- Five δ_z parameters were dropped from the compression because they are prior-dominated. The summary stays 7-D, but δ_z is still varied in the sims.
- Two-pass compression: re-run the compression at the MAP from the first pass, keeping the same sims.
- GoF: χ² of the compressed summary vs 500 sims at the MAP (Gelman 1996 style). Unconstrained parameters are fixed at the compression fiducial, not the MAP, because the MAP wanders for them.
- The noise-bias estimator uses one random rotation with a (2ℓ+1)-weighted mean over ℓ = 200–1500, instead of hundreds of rotations. This is what made ~10⁴ sims affordable.
- Coverage was run on a narrower S8 range than the prior, to focus on the region of interest.
- Blinding caught a mismatch between the measurement codes for sims and data, which initially gave an "anomalously low" S8. Sim/data pipeline parity is a real failure mode.
- NDEs are stacked with weights from validation loss.

## Caveats

- The 2-pt summary means the "SBI" mainly relaxes the Gaussian likelihood. Most parameters are prior-dominated; only S8 and A_IA are constrained.
- Coverage was checked on only 2 parameters.
- Log-normal fields limit higher-order accuracy.
- The GoF is at the MAP, not posterior-predictive.
