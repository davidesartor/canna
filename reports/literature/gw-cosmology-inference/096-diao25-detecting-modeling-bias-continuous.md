# Detecting Modeling Bias with Continuous Time Flow Models on Weak Lensing Maps

Diao, Dai, Seljak et al., 2025, JCAP 08 (2025) 004 · [arXiv:2505.00632](https://arxiv.org/abs/2505.00632)

**Evidence: B.** Peer-reviewed (Journal of Cosmology and Astroparticle Physics); 4 citations in
1.5 yr (Semantic Scholar, 2026-10-19).

## What they do

Forward-model validation recast as OoD detection: test statistic t(x) = log p(x) evaluated at the field level with an unconditional continuous-time flow (diffusion VP-SDE β_t = 20t, or OT flow matching x_t = (1−t)x0 + tε), density via the instantaneous change-of-variables integral with Hutchinson trace (Gaussian probes), Euler 1000 steps. Compared against feature-level NFs (RealNVP on scattering-transform coefficients or on a VMIM-trained ResNet-34 compression) and a field-level multiscale NF.

## What they find

DMO vs baryon-corrected (BCM) WL convergence maps at fixed cosmology, 128² maps (3.5°×3.5°, shape noise n_g = 30/50/100): AUROC OTFM 0.87/0.92/0.95, diffusion 0.85/0.90/0.94, ST+NF 0.73/0.77/0.81, CNN(VMIM)+NF 0.54/0.55/0.60, multiscale field NF ≳0.65. Across 7 cosmologies OTFM AUROC 0.79–0.93 (worse when fluctuation amplitude is low). Higher resolution: 0.87 → 0.94; 60 deg² (5 maps' likelihoods summed): perfect separation. GRF sanity: OTFM log p matches analytic with "relative scatter ~6" (Euler, few Hutchinson probes). Model selection: |log p(x|M) − E_InD log p| correctly picks the true shape-noise level among n_g ∈ {25,30,35} for all 304 mocks, though raw log p overlaps. GLOW assigns HIGHER likelihood to smoothed/scaled/masked OoD maps; OTFM correctly assigns lower.

## Implementation details

Use the *typicality* deviation |log p − E[log p]|, not raw log p, for model selection. Score maps in 64² patches, sum log-densities (ignoring correlations) to scale up. A cosmology-optimised (VMIM) compressor is nearly blind to misspecification — it discards non-cosmological features — so detection needs a non-sufficient statistic or the full field. Unconditional training because only ~80 cosmologies available. U-Net with 36 res-blocks enc/dec, attention below 16².

## Caveats

Idealised (posterior replaced by δ at true cosmology for feature methods), single systematic type (BCM), small maps; density accuracy itself is rough (Euler + few probes); no posterior inference performed.
