# Strictly Proper Scoring Rules, Prediction, and Estimation

Gneiting & Raftery, 2007, J. Am. Statist. Assoc. 102(477), 359–378 · [DOI:10.1198/016214506000001437](https://doi.org/10.1198/016214506000001437)

**Evidence: A.** Peer-reviewed (JASA); 7468 citations in 19.6 yr (Semantic Scholar,
2026-10-22).

## What they do

This is a review and theory paper on proper scoring rules S(P, x), here positively oriented (higher is better). S is proper if E_Q S(Q, X) ≥ E_Q S(P, X) for all P, Q. It is strictly proper if equality holds only for P = Q. Every proper rule comes from a convex entropy G(P) = S(P, P). Its divergence d(P, Q) = S(Q, Q) − S(P, Q) ≥ 0 is the expected penalty for quoting P when the truth is Q (Theorem 1).

The paper introduces the **energy score** and the general **kernel score** family. Both are defined through expectations under P, so they can be estimated from samples of P alone:

- CRPS: CRPS(F, x) = ½ E|X − X′| − E|X − x| (Eq. 21).
- Energy score on ℝ^m, β ∈ (0, 2): ES(P, x) = ½ E‖X − X′‖^β − E‖X − x‖^β (Eq. 22). It is strictly proper for P with finite β-moment (Székely 2003). At β = 2 it degenerates to −‖μ_P − x‖², which is proper but *not strictly proper*: it sees only the mean.
- Kernel score (Theorem 4): for any non-negative continuous negative-definite kernel g, S(P, x) = ½ E g(X, X′) − E g(X, x) is proper. Examples are the Brier score, the CRPS (g = |x − x′|), the energy score (g = ‖x − x′‖^β), and a circular CRPS with g = angular distance (Example 10). Non-Euclidean energy scores use ‖·‖_α^β under Assumption 1 (m ≥ 2: α ∈ (0, 2], β ∈ (0, α]). Theorem 5 gives strict propriety for g = ψ(‖x − x′‖²) with −ψ′ completely monotone and not constant.
- Positive-definite / characteristic-function form: S(P, x) = −∫ |φ_P(y) − e^{i⟨x,y⟩}|² μ(dy) (Eq. 39). It is strictly proper if μ is finite and charges all intervals. The weight ‖y‖^{−m−β} recovers the energy score.
- Rules that depend on moments only: the Dawid–Sebastiani score −log det Σ_P − (x − μ_P)ᵀ Σ_P⁻¹ (x − μ_P) (Eq. 25) is proper, and strictly proper only within moment-determined families. The PMCC −(x − μ)² − σ² (Eq. 26) is *improper*: it rewards quoting a point mass.
- Quantile and interval scores. The pinball loss is (x − r)(1{x ≤ r} − α). The interval score for a central (1 − α) interval [l, u] is S_α^int = (u − l) + (2/α)(l − x)1{x < l} + (2/α)(x − u)1{x > u} (Eq. 43). It rewards narrow intervals and penalises misses.

## What they find

- **Interval score vs coverage** (bilinear process, 10⁵ one-step 95% intervals, Table 2). The true conditional interval I has coverage 95.01%, width 4.00 and interval score 4.77. The *unconditional/stationary* interval J has 95.08%, 5.45 and 8.04. The minimum-width interval K has 94.98%, 3.79 and 5.32; it collapses when the variance is highest. All three have nominal coverage. Only the proper score ranks the truth first and gives the unconditional forecast the worst score.
- **Propriety matters** (sea-level pressure, 5-member ensemble, 16 015 forecast cases, Gaussian forecast N(ensemble mean, (rσ)²), Table 3). The proper scores are maximised at inflation r > 1: QS 2.18, SphS 1.84, LogS 2.41, CRPS 1.62. The improper linear and probability scores are maximised at r = 0.05 and 0.02, i.e. they reward near-deterministic forecasts. The RMSE/spread ratio is r₀ = 1.55. Nominal coverage of the 50%/90% intervals is reached at r = 1.78/2.11, while the interval score is optimal at r = 1.56/1.72.
- LogS chose the largest r because it heavily penalises a few low-spread cases where the ensemble variance collapses. The CRPS is more robust to such outliers.
- **Skill scores are generally improper**, (S_fcst − S_ref)/(S_opt − S_ref), even when S is proper. A climatological reference is "calibrated by construction, but often lack[s] sharpness".

## Implementation details

- Sample estimator. With m draws X_1..X_m from P and observation x, plug in ES ≈ (1/(2m²)) Σ_{i,j} ‖X_i − X_j‖^β − (1/m) Σ_i ‖X_i − x‖^β, at O(m²·d) cost. The paper notes the CRPS of an m-sample can be computed from order statistics in O(m log m) (Hersbach 2000). It does not discuss the bias of the plug-in. The V-statistic includes the i = j zeros, which shrinks the spread term. The "fair" U-statistic with 1/(m(m−1)) over i ≠ j is unbiased for E‖X − X′‖^β; it comes from later literature, not this paper.
- The divergence of a kernel score is d(P, Q) = E g(X, Y) − ½ E g(X, X′) − ½ E g(Y, Y′) (Hoeffding-type inequality, Eq. 31). For the energy score this is half the energy distance between forecast and truth. A mean score gap between two forecasters therefore estimates a difference of energy distances to the truth.
- Ranking. Average the score over the *same* set of cases. Scores over different case sets are not comparable, because intrinsic predictability varies.
- Optimum-score estimation: fitting parameters by maximising a mean strictly proper score is consistent (Eq. 59). MLE is the special case of the log score, and minimum-CRPS fitting is a robust M-estimator.
- Choice of β: β → 2 loses strict propriety. β = 1 is the usual CRPS generalisation. The scores are not affine-invariant, so the coordinate scaling matters.

## Caveats

- Theory and review. No power study of the energy score or other multivariate scores for detecting specific misspecifications (dependence, dispersion).
- Formal tests of score differences are only pointed to (Diebold & Mariano 1995 and others), not developed.
- No treatment of finite-ensemble bias or of fair scores.
- Full text read (author PDF).
