# Diffusion Models in Simulation-Based Inference: A Tutorial Review

Arruda et al., 2025, TMLR · [arXiv:2512.20685](https://arxiv.org/abs/2512.20685)

**Evidence: B.** Peer-reviewed; 17 citations in 0.8 yr (Semantic Scholar, 2026-10-19).

## What they do

- A tutorial that puts score / DM / FM / consistency-model posterior estimators in one frame (z_t = α_t z_0 + σ_t ε).
- It then catalogues SBI adaptations:
  - Guidance: CFG-style tempering, constraint guidance, simulator-feedback control.
  - Compositional score aggregation: no/complete/partial pooling, with GAUSS/JAC precision-weighted sums.
  - Structured targets: multiscale wavelets, causal DAG flows, Simformer joint masking, Riemannian scores.
- It closes with a four-case-study design benchmark in BayesFlow.
- It notes that the parameterisations (ε, z_0, v, F, score) are equivalent up to loss weighting. Switching target without reweighting silently changes the effective log-SNR weighting.

## What they find

- **Case 1 (SBIBench, 10 tasks, C2ST):**
  - VP-EDM with F-prediction is best; cosine-VP is close; VE is worse.
  - FM is competitive, slightly worse on the smallest tasks.
  - A power-law t-sampling with ρ=-0.6 (p(t) ∝ (1-t)^{1/(1+ρ)}, upweighting near data) helps in most tasks, notably Lotka-Volterra.
  - OT-CFM: no clear gain. Computing the OT plan while ignoring the condition hurts.
  - EMA: no systematic gain.
  - Annealed Langevin, even at 4× the steps, and predictor-corrector both do not help.
  - Adaptive two-step SDE solver: fastest and most accurate.
  - DiT vs residual MLP: about equal, and the DiT is 3.5× slower.
- **Case 2 (Beer ODE, 72 parameters, 1000 test sets, 10 seeds), Table 4:**
  - EDM-VP-F-ODE: NRMSE 0.42, calibration error 0.03, contraction 0.68, C2ST 0.57, rank 1.
  - FM uniform: NRMSE 0.39 (best), calibration error 0.07, contraction 0.77 (overconfident direction), rank 10.
  - FM with ρ=-0.6: calibration error 0.05, TARP 0.01, rank 7.
  - Continuous CM: NRMSE 0.70, calibration error 0.14.
  - ε-pred with Euler-ODE collapses: NRMSE 1.38, contraction 0.01.
  - MCMC: NRMSE 0.18 but calibration error 0.29, because it fails to mix. It takes 3 days for one dataset, versus under 1 day for all amortised training plus 1000 inferences.
- **C2ST in high dimension:**
  - At 72-D it stays near chance even with log-likelihood summaries.
  - With ε-prediction it reads 0.52 while the posterior is barely updated from the prior.
  - ECE and TARP agree with each other.
- **Case 3 (GRF):**
  - DM is best calibrated at high resolution, with FM second; CMs are miscalibrated.
  - Online simulation improves calibration for DM and FM.
- **Case 4 (hierarchical):**
  - Composition with damping d_0 < 1 improves calibration but costs sharpness.
  - Mini-batching the composed groups is harmless.

## Implementation details

- Compute the loss in the ε/noise domain regardless of the predicted target, to keep the weighting fixed.
- Condition the network on normalised λ_t rather than t.
- Truncate t to [t_0, t_1], because the Lipschitz norm of the velocity spikes near the boundaries.
- Use adaptive higher-order ODE solvers (fewer, larger steps beat Euler).
- ODE collapse errors, where a deterministic flow misses the typical set, can be diagnosed with SBC/calibration. SDE sampling helps slightly in low dimension only.
- Use a metrics suite rather than C2ST alone: NRMSE, ECE, contraction, C2ST, TARP. Rank by the sum of standardised metrics.
- Guidance biases the posterior; correct it with Langevin/SMC if exactness matters.
- Recommendation (Fig. 16): VP-EDM DM with an adaptive SDE solver, or FM with an adaptive higher-order ODE solver. Use CMs only when latency dominates.

## Caveats

- Benchmarks are Euclidean, with ≤72 parameters and no periodic or spherical coordinates.
- FM is run only in its plain linear-Gaussian form, with no prior-base or Riemannian variant.
- It is a review, so section 4 restates other papers. The section 6 numbers are its own contribution.
- All runs train with the noise-prediction loss (10), so the FM-vs-DM comparison partly confounds weighting with family.
