# SimBIG: The First Cosmological Constraints from Non-Gaussian and Non-Linear Galaxy Clustering

Hahn et al., 2023, arXiv/PRD letter · [arXiv:2310.15246](https://arxiv.org/abs/2310.15246)

**Evidence: B.** Peer-reviewed; 0 citations in 3.0 yr (Semantic Scholar, 2026-10-19).

## What they do

Compare SimBIG NPE posteriors from the BOSS CMASS-SGC (~10% of the BOSS volume) across three summaries:
- P_ℓ;
- the reduced bispectrum monopole Q0 (10,052 triangles to k = 0.5);
- a 3D CNN on a 64×64×128 CIC mesh (k < 0.28), trained by MSE to regress θ, with SWA.
The flows are MAF/NSF from `sbi`, with architecture chosen per summary. Posterior robustness is demanded on out-of-distribution forward models.

## What they find

- Against SimBIG P_ℓ, B0 tightens Ω_b, h, n_s, Ω_m and σ8 by 2.78×, 3.17×, 1.91×, 1.77× and 1.53×. The CNN tightens them by 2.51×, 2.51×, 1.46×, 1.42× and 1.75×.
- Against PT P_ℓ, the best of B0/CNN is 1.6/1.5/1.7/1.2/2.3× tighter. S8 is ~1.9× tighter (equivalent to 4× the volume) and H0 1.5×.
- S8 = 0.757 (−0.053/+0.048) from B0.
- Robustness ablation: if the out-of-distribution robustness requirement is relaxed, the CNN is >2× more precise, but biased on the alternative-forward-model test sets.
- WST reaches B0-level precision but fails robustness, so it is excluded.

## Implementation details

- Test sims come in three sets at a fiducial cosmology: 500 from the same forward model, 500 with a different halo finder and HOD, and 1,000 with a different N-body code and halo finder. The posterior must stay consistent across all three.
- The CNN is regularised to pass that test with dropout, weight regularisation and SWA (weight marginalisation). This deliberately "discards a substantial amount of cosmological information".
- Q0 (the normalised bispectrum) is used instead of B0 to reduce the dynamic range of the input.
- Not forward-modelled systematics are handled with data weights.
- Did not work: the full-power CNN and the WST. Both latch onto imprints specific to the forward model.

## Caveats

- The 2,000-cosmology training set is small.
- The robustness criterion is qualitative.
- The CNN gives Ω_m 1.07× broader than PT.
