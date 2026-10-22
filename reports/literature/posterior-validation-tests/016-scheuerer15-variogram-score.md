# Variogram-Based Proper Scoring Rules for Probabilistic Forecasts of Multivariate Quantities

Scheuerer & Hamill, 2015, Mon. Weather Rev. 143(4), 1321–1334 · [DOI:10.1175/MWR-D-14-00269.1](https://doi.org/10.1175/MWR-D-14-00269.1)

**Evidence: A.** Peer-reviewed (Monthly Weather Review); 222 citations in 7.6 yr (Semantic
Scholar, 2026-10-22).

## What they do

The energy score (ES) is the standard multivariate proper score for ensembles. Earlier work (Pinson & Girard 2012, Pinson & Tastu 2013) had found it barely sensitive to wrong correlations between components. The Dawid–Sebastiani score (DSS), log det Σ_F + (y − μ_F)ᵀ Σ_F⁻¹ (y − μ_F), does see correlations, but it needs a covariance matrix. Estimated from a small ensemble, that matrix makes the score unreliable.

The paper proposes the **variogram score of order p** (VS-p, negatively oriented):

S_p(F, y) = Σ_{i,j=1}^d w_ij ( |y_i − y_j|^p − E_F|X_i − X_j|^p )² (Eq. 3).

It compares the observed pairwise component differences with the forecast's expected ones. With p = 2 and Gaussian components, E|Y_i − Y_j|² = (μ_i − μ_j)² + σ_i² − 2σ_iσ_jρ_ij + σ_j² (Eq. 2). So the score depends on the means, the variances and the correlations.

## What they find

- **Setup.** n = 5000 observation vectors, ensembles of m = 20 or 100, and each experiment repeated 10 times. Discrimination is judged by whether the box plots of the 10 mean scores for the correct and the wrong forecast overlap. There is no formal test.
- **Marginal errors** (d = 5, Gaussian AR(1) with corr = exp(−|i − j|/3)). The forecasts tested had a linear mean trend from −0.5 to 0.5, variance 1.5 (overdispersed) and variance 0.667 (underdispersed). The ES detects the mean trend best. It separates the over- and underdispersed forecasts from the correct one "less distinctly". VS-0.5 detects dispersion errors better than the ES, but the mean error less clearly. The ES barely changes between m = 20 and m = 100. The authors read this as a sign that the ES cannot tell a good sample representation from a sparse one. The VS-p improves with m.
- **DSS from a 20-member ensemble ranks wrongly.** In the marginal experiment it prefers the overdispersed ensemble to the calibrated one. With m = 100 it ranks correctly.
- **Correlation strength** (forecast range r = 2, 3 or 4.5 against a true r = 3; d = 5 and 15). The ES "can hardly discriminate", confirming Pinson & Tastu (2013). VS-0.5 and VS-1 separate the cases well. VS-2 does better than the ES but is noisy. Going from d = 5 to d = 15 slightly *hurts* the VS-p, because more weakly correlated pairs add variance.
- **Correlation model** (d = 15; long-range or periodic correlation against an exponential one with the same lag-1 value). The ES is insensitive again. A smaller p gives better discrimination.
- **Same first two moments, different process** (shot noise against a Gaussian with matched mean, variance and correlation). Only VS-0.5 discriminates well. The DSS prefers the wrong model even at m = 100.
- **Weights** (VS-0.5, d = 15, m = 20). Localised weights w_ij = [1 − (|i − j|/3)²]₊ discriminate best for the range errors, and uniform weights for the long-range error. Inverse-distance weights are a compromise, and even uniform weights beat the ES.
- **The transformation does the work.** Applying the ES, MAE, MSE or mean CRPS to the transformed vector of weighted |x_i − x_j|^0.5 values all detect the wrong correlation model.
- **Wind case study** (5 Colorado sites, 11-member GEFS reforecast, 2013, lead times 24/48/72 h; Table 1, skill against the raw ensemble). At 48 h: ECC-Q has ESS 0.119, VSS-0.5 0.119, VSS-1 0.096. Random-Q (calibrated margins, independent components) has ESS 0.108, VSS-0.5 −0.020, VSS-1 −0.017. At 72 h: ECC-Q 0.063/0.036/0.027 and random-Q 0.051/−0.087/−0.063. The ES almost cannot tell the ensemble with the right dependence from the one with independent components. The VS gives the independent one negative skill.

## Implementation details

- **Sample estimator.** E_F|X_i − X_j|^p ≈ (1/m) Σ_k |x_i^(k) − x_j^(k)|^p (Eq. 4). The cost is O(m·d²), with no m² term, so it is cheaper than the ES's O(m²·d). The estimator is unbiased for the forecast variogram. The squared difference in Eq. 3 still picks up a positive bias of var(·)/m. That is why the scores improve with m. In their tests this sampling bias never changed the ranking, while the DSS's did.
- **Propriety.** The score is proper for distributions with finite 2p-th moments, since the mean minimises the expected squared error for each pair. It is **not strictly proper**: it sees only the p-th absolute moment of each pairwise difference.
- **Blind spots.** It cannot see a bias common to all components or a random shift common to all of them. Some combinations of errors cancel. For example, an erroneous trend and underdispersion cancel at p = 2, and too-strong correlations cancel against inflated variances. So check the marginals first with univariate scores (CRPS, PIT), then run the VS-p.
- **Choice of p.** p = 0.5 works best: |X_i − X_j|^0.5 is nearly symmetric for Gaussian data, so it has low sampling variance. p = 2 is clearly worse. If the components are skewed, an even smaller p may help.
- **Weights.** Downweight pairs that are expected to be weakly correlated (higher variance, low signal). They suggest inverse distance by default, localisation when the correlation scale is known, and weights summing to one per case if d varies.
- **Mixed units.** Standardise each component by its climatological mean and standard deviation first, since the score subtracts components from each other.
- **Observation error.** Add simulated error draws ε to the ensemble members inside the expectation (the Section 5 formula).
- The wind comparison uses skill scores relative to the raw ensemble. No significance test is reported.

## Caveats

- Discrimination is judged visually from 10 repetitions. No power curves, no Diebold–Mariano tests, no error rates.
- All the examples are smooth stationary processes with a natural distance between components. For unordered components the weights are ad hoc.
- The DSS comparison uses the plug-in sample covariance with no shrinkage.
- The finding that the ES is insensitive to dependence is contested. Ziel & Berk (2019, arXiv:1910.07325, unreviewed) argue that it rested on relative score changes. With Diebold–Mariano tests they find the ES "comparably well" at detecting dependence errors. Alexander et al. (arXiv:2101.12693) again find VS-0.5 more discriminating than the ES. The Pinson & Tastu (2013) DTU technical report is not peer-reviewed, and its full text was not read here.
- Full text read (NOAA repository PDF).
