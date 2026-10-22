# SimBIG: A Forward Modeling Approach To Analyzing Galaxy Clustering

Hahn et al., 2022, PNAS · [arXiv:2211.00723](https://arxiv.org/abs/2211.00723)

**Evidence: B.** Peer-reviewed; 1 citation in 4.0 yr (Semantic Scholar, 2026-10-19).

## What they do

Forward-model BOSS CMASS-SGC. The pipeline is Quijote N-body → Rockstar halos → 9-parameter HOD (assembly, concentration and velocity bias) → survey geometry, masks and 60% fiber collisions. A MAF NPE is trained (`sbi` package) on 20k catalogues of P_ℓ(k) + n̄_g. The posterior covers 5 ΛCDM + 9 HOD + 1 A_shot parameters.

## What they find

- σ8 = 0.812 (−0.068/+0.067), 27% tighter than the PT analysis at k_max = 0.25 and 13% tighter than the emulator analysis.
- Ablation: SimBIG run to k_max = 0.25 gives σ8 = 0.861 (−0.091/+0.070), the same width as the emulator. So the gain comes from k > 0.25.
- Ω_m is broader than PT because there are only 2,000 distinct cosmologies.
- Re-calibration with test sims gives Ω_m = 0.284 (−0.017/+0.021), against 0.292 (−0.040/+0.055) uncalibrated: a ~2.5× potential gain from more sims.
- Inconsistency: the Conclusions quote different numbers (Ω_m = 0.316, σ8 = 0.668) from the Results section, apparently an error in v1.

## Implementation details

- Training set: 2,000 N-body sims × 10 HOD draws = 20k, arranged as a Latin hypercube (cheap nuisance resampling on expensive cosmology sims).
- Data split 90/10. Adam at lr 5e-4. Early stopping after 20 epochs without improvement in validation log-likelihood.
- MAF with 6 MADE blocks, each with 9 hidden layers and 186 units, "through experimentation".
- Systematics that are not forward-modelled (redshift failures, stellar density, seeing) are left as weights on the data.
- Failure mode: a limited number of distinct θ makes the NPE conservative (too wide), not biased.

## Caveats

P_ℓ only; no coverage test in this paper; the numbers are internally inconsistent across sections.
