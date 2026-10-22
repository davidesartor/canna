# Galaxy Clustering Analysis with SimBIG and the Wavelet Scattering Transform

Régaldo-Saint Blancard et al., 2023/v2 2024, PRD · [arXiv:2310.15250](https://arxiv.org/abs/2310.15250)

**Evidence: B.** Peer-reviewed (Physical Review D); 18 citations in 3.0 yr (Semantic Scholar,
2026-10-19).

## What they do

- Hand-designed reduced 3D WST: 120 oriented spline wavelets (J = 6, Q = 2, 10 dodecahedron directions, σ_ϑ = π/4).
- Moments S0, S1, S2 with exponents p, log-normalised by the previous order (S̄1 = log(S1/S0), S̄2 = log(S2/S1)).
- Averaged over azimuth about the line of sight (redshift-space symmetry), giving 3,675 coefficients plus n_gal and P_shot.
- Fed directly (no compression) to an ensemble of MAF NPEs trained on 20k SimBIG catalogues.

## What they find

- SBC on the validation set: Ω_b, h and n_s are calibrated. Ω_m and σ8 are under-confident (∩-shaped ranks), which the authors blame on the limited training set.
- Ensembling ablation: the individual top-10 models are slightly over-confident on Ω_b, h and n_s. The uniform 10-model mixture fixes it.
- Mock challenge: σ8 is biased low on TEST2 (AbacusSummit) and high on TEST1 (FoF). The other parameters are robust.
- The fix is a scale cut j_min = 2, which drops k ≳ 0.3. The σ8 (μ − θ)/σ histograms then agree, and the other parameters are unaffected.
- With j_min = 2 against PT P_ℓ, the 68% intervals are 1.2/1.5/0.8/1.3/1.8× tighter for Ω_m, Ω_b, h, n_s and σ8.
- Unresolved concern: ensemble members with similar validation scores, all passing SBC, give highly variable BOSS posteriors. The authors read this as model misspecification (the data is an outlier) and raise the ensemble to 50 for j_min = 2. They refuse to interpret the cosmology.

## Implementation details

- ~2,000-model WandB random search over several successive narrowed sweeps.
  - Final ranges: 5–11 transforms, 256–1024 hidden units, 2–4 blocks, dropout 0.1–0.2, batch 20–100, lr 5e-6 to 5e-5.
  - Winners: lr 5e-6 to 1.2e-5, mostly 2 blocks, dropout 0.1. Note the very small learning rates.
- BatchNorm in the MAF conditioner MLP "significantly improved" the validation score.
- Did not work: NSF overfit in a way standard regularisation could not fix, so MAF was kept.
- Log-ratio normalisation of the hierarchical coefficients evens their magnitudes.
- SBC uses m = 2,500 posterior samples per validation sim. The validation set was also the early-stopping set, since there was no separate test set.

## Caveats

- The training set is not i.i.d. (10 HOD draws per N-body).
- SBC is run on the early-stopping set.
- Posterior variability across ensemble members on real data is unexplained.
- The scale cut was tuned on the test sets.
