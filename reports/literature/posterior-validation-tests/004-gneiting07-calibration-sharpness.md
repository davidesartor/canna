# Probabilistic Forecasts, Calibration and Sharpness

Gneiting, Balabdaoui & Raftery, 2007, J. R. Statist. Soc. B 69(2), 243–268 · [DOI:10.1111/j.1467-9868.2007.00587.x](https://doi.org/10.1111/j.1467-9868.2007.00587.x)

**Evidence: A.** Peer-reviewed (JRSS-B); 2024 citations in 19.5 yr (Semantic Scholar,
2026-10-22).

## What they do

Nature picks a distribution G_t, the forecaster issues a predictive CDF F_t, and x_t ~ G_t is observed. The "ideal" forecaster has F_t = G_t. The paper shows that PIT uniformity cannot tell the ideal forecaster from worse ones. It proposes the paradigm "maximise the sharpness of the predictive distributions subject to calibration". Calibration is a joint property of forecasts and outcomes. Sharpness is a property of the forecasts alone.

They define three modes of calibration, as T → ∞:
- **Probabilistic:** (1/T) Σ G_t ∘ F_t⁻¹(p) → p for all p. Equivalent to uniform PIT values p_t = F_t(x_t) (Theorem 2, under mixing).
- **Exceedance:** (1/T) Σ G_t⁻¹ ∘ F_t(x) → x for all x. No obvious sample analogue exists.
- **Marginal:** the average forecast CDF equals the average true CDF (forecast "climate" = observed climate). Checked by comparing F̄_T(x) = (1/T) Σ F_t(x) with the empirical CDF of the x_t (Theorem 3).

The three modes are logically independent and occur in every combination (Table 2).

## What they find

- Simulation: G_t = N(μ_t, 1), μ_t ~ N(0, 1), 10 000 cases. The four forecasters are ideal N(μ_t, 1); climatological N(0, 2), i.e. the unconditional marginal; unfocused ½{N(μ_t,1) + N(μ_t+τ_t,1)} with τ_t = ±1; and Hamill's, N(μ_t+δ_t, σ_t²) with (δ, σ²) ∈ {(½,1), (−½,1), (0,169/100)}. All four PIT histograms are essentially uniform (Fig. 1).
- Central-interval coverage, 50%/90% (Table 3): ideal 51.2/90.0, climatological 51.3/90.7, unfocused 50.1/90.1, Hamill's 50.9/89.5. Coverage cannot rank them.
- Average interval width, 50%/90% (Table 4): ideal 1.35/3.29, Hamill's 1.49/3.62, unfocused 1.52/3.68, climatological 1.91/4.65.
- Average scores (Table 5), LogS / CRPS: ideal 1.41/0.56, Hamill's 1.52/0.61, unfocused 1.53/0.63, climatological 1.75/0.78. Both proper scores rank the ideal forecaster first and the climatological one last.
- The climatological forecaster is probabilistically *and* marginally calibrated but not exceedance calibrated. Hamill's forecaster violates probabilistic calibration by at most |ε(p)| ≤ 0.0032, which is invisible in a histogram.
- Theorem 1 is a weak sharpness principle. A constant forecast F that is probabilistically calibrated has var(F) ≥ (1/T) Σ var(G_t), with equality only if all E(G_t) are equal. So a calibrated climatological forecast is never sharper than the truth on average. Counter-examples show that a non-constant forecaster can be probabilistically, exceedance or marginally calibrated and still be *sharper* than the ideal one. Calibration alone does not bound the width from below in general.
- Wind case study (Stateline, 2-h-ahead hourly wind speed, 5136 forecasts, May–Nov 2003). The RST method has CRPS 0.97 m/s, against 1.12 for autoregressive and 1.20 for persistence. Its 90% intervals are 5.31 m/s wide on average, against 6.55 and 7.51. The AR PIT histogram is hump-shaped (overdispersed: 55.6% coverage at nominal 50%). RST beats AR in all 7 months. Under equal skill that has probability (½)⁷ = 1/128, a sign test on monthly blocks.

## Implementation details

- **PIT/rank histogram.** p_t = F_t(x_t). For a sample (ensemble), the rank of x_t among the pooled ensemble values gives the analogue; the authors suggest the same for MCMC output. Use 10–20 bins. Hump = too wide. U = too narrow. Triangle = biased.
- **Marginal calibration plot.** Plot F̄_T(x) − Ĝ_T(x) against x, or the difference of the quantile functions against q. Under marginal calibration both fluctuate around 0.
- **Sharpness diagram.** Box plots (5th, 25th, 50th, 75th, 95th percentiles) of the widths of the central 50%/90% intervals across cases. The average width is not enough when the width varies by case.
- **CRPS from samples.** crps(F, x) = E|X − x| − ½ E|X − X′|, with X, X′ ~ F i.i.d. It is in the units of x and reduces to the absolute error for a point forecast. Averaged over cases, CRPS = ∫ BS(y) dy, where BS(y) is the Brier score of the threshold event {x ≤ y}. Plotting BS(y) against y shows where along the axis a forecaster loses.
- **Comparisons.** Compare mean scores. For tests, use blocks that are independent in practice (here months) or Diebold–Mariano-type tests, and take care that score differentials are not serially dependent.
- **Time-series checks.** Ideal k-step-ahead PITs are at most (k−1)-dependent. Check the ACF of the PIT, (PIT − ½)² and (PIT − ½)³, or of Φ⁻¹(PIT).

## Caveats

- Univariate only. Multivariate and joint calibration are not treated.
- The "sharpness principle" (ideal ⇔ maximal sharpness subject to calibration) is a conjecture. It is proven only for constant forecasts (Theorem 1).
- Formal uniformity tests are cited but not developed. The authors warn that dependent or spatially pooled PITs invalidate them.
- The CRPS sample formula is given. Its finite-ensemble bias (the ½E|X − X′| term with m members) is not discussed.
- Full text read (author PDF).
