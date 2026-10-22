# Assessing the Calibration of High-Dimensional Ensemble Forecasts Using Rank Histograms

Thorarinsdottir, Scheuerer & Heinz, 2016, J. Comput. Graph. Statist. 25(1), 105–122 · [DOI:10.1080/10618600.2014.977447](https://doi.org/10.1080/10618600.2014.977447) · [arXiv:1310.0236](https://arxiv.org/abs/1310.0236)

**Evidence: B.** Peer-reviewed (JCGS); 83 citations in ~10 yr (Semantic Scholar,
2026-10-22).

## What they do

The paper proposes two pre-rank functions for the two-step multivariate rank histogram of Gneiting et al. (2008), meant to keep their power in high dimension. S = {x_1, …, x_m} holds m − 1 ensemble members plus the observation x_m. Each element gets a pre-rank. The observation's rank is the rank of its pre-rank, with ties broken at random. rank_S(x_k) = Σ_i 1{x_ik ≤ x_k} is the univariate rank of component k.

- **Band-depth pre-rank** (the modified band depth of López-Pintado & Romo 2009, with n = 2): the average over components of the number of member pairs whose range contains x_k,
  ρ_bd(x) = (1/d) Σ_k [ rank_S(x_k)(m − rank_S(x_k)) + (rank_S(x_k) − 1) Σ_i 1{x_ik = x_k} ] (Eq. 3).
  Without ties it is (1/d) Σ_k (m − rank_S(x_k))(rank_S(x_k) − 1) + (m − 1) (Eq. 4). It orders from the centre outward: central elements get high pre-ranks.
- **Average-rank pre-rank**, ρ_a(x) = (1/d) Σ_k rank_S(x_k) (Eq. 5). It reduces to the usual rank when d = 1.

Both are compared with the multivariate rank of Gneiting et al. (2008), ρ(x) = Σ_i 1{x_i ⪯ x}, and with the minimum spanning tree (MST) pre-rank, the length of the MST of S \ x (Smith & Hansen 2004; Wilks 2004). If the m elements are i.i.d., every one of these ranks is uniform.

**How to read the histograms.**
- *Band depth.* Too many high ranks means the ensemble is overdispersed (the observation is too central). Too many low ranks means it is underdispersed or biased. Correlations that are too weak in the forecast give a ∪ shape, and correlations that are too strong give a ∩ shape. It cannot show the sign of a bias.
- *Average rank.* It reads like a univariate histogram: ∪ for underdispersed, ∩ for overdispersed, triangular for a constant bias. Correlation that is under- or overestimated looks like over- or underdispersion. Opposite biases in different components can cancel.

## What they find

- **Shapes** (d = 3, 19 members, observation N(0, I), forecast N(μ, σ²I) with μ ∈ {0, 1} and σ ∈ {0.5, 1, 2}, 10 000 repetitions; Figs 2–3). Both new pre-ranks show the shapes described above.
- **Dimension** (d = 5 and 15, 19 members, σ = 0.5 or 2; Figs 4–5). The multivariate rank degenerates. At d = 5 only the upper half of its histogram shows miscalibration. At d = 15 it looks near-uniform even though the forecast is severely miscalibrated. The average-rank histograms are unchanged from d = 3. The band-depth evidence of miscalibration *grows* with d. The MST histograms are nearly identical to the band-depth ones. A larger ensemble generally makes miscalibration easier to see (results not shown), but it does not fix the multivariate rank's degeneracy.
- **Correlation strength** (Gaussian AR(1) with Cov = exp(−|i − j|/τ), τ = 3 for the truth and τ = 2 or 4 for the forecast, d = 5, m = 20; Fig. 6). The margins are exactly calibrated, so no univariate check can see this error. All four methods detect it. The multivariate rank's left side flattens out. Band depth and average rank give ∪ when the forecast correlations decay too fast and ∩ when they decay too slowly. MST gives "the clearest indication".
- **Mechanism** (Tables 1–2, τ = 3 vs τ = 2, 30 000 repetitions). The mean ranks of the observation and of a random member agree, e.g. 10.5 vs 10.5 at m = 20, d = 5. The rank *variances* differ: at m = 20 the observation's is 37–40 against the member's 33. At m = 500 it is ≈ 23 400–25 600 against ≈ 20 700–21 000. Dimension, from d = 5 to 500, has only a minimal effect. Closed-form pre-rank variances for the extreme case of a fully dependent observation and independent forecasts are Eqs. 7–10.
- **Correlation shape** (d = 15, m = 20; Fig. 7). Model a) is a damped cosine around the exponential: the average rank is flat, while MST detects the error well and band depth detects it too. Model b) has long-range dependence: the average rank is clearest and MST is almost flat. Model c) has zero correlation beyond lag 5: the average rank is slightly ∩ and band depth slightly ∪. The average rank is more sensitive to long-lag correlation; band depth and MST to short-lag correlation.
- **Case study.** ECMWF 50-member temperature forecasts for Berlin Tegel, 12 lead times from 6 to 72 h, 823 days. The post-processing is bias correction plus error dressing, with errors drawn (i) independently per lead time, (ii) with ECC or (iii) from a multivariate normal. All three are nearly calibrated per lead time (Fig. 8). With independent errors, the average rank is ∪, band depth is skewed low, and MST and the multivariate rank are skewed high. On any day at least half the curves had multivariate pre-rank 1. ECC is "somewhat too outlying". The multivariate normal is close to uniform.

## Implementation details

- **Samples only.** Every pre-rank uses only the members and the observation. Band depth, average rank and the ⪯ rank depend only on per-component ranks, so they are invariant to per-component normalisation. MST uses Euclidean distances, so it needs the components on comparable scales.
- **Cost per case.** Average and band-depth ranks need only the d univariate rankings, O(d·m log m). The multivariate rank is O(m²·d). MST needs m minimum spanning trees on ≤ m points, roughly O(m³·d) with dense Prim.
- **Ties.** Breaking ties at random is what flattens the degenerate multivariate rank. The new pre-ranks produce far fewer ties.
- **Use marginal checks first.** Univariate calibration rules out compensating effects, e.g. between the marginal variances and the correlations.
- **No single best pre-rank.** Each projects onto one univariate aspect and loses the others. They recommend looking at several kinds of histogram.
- **Tests.** Formal uniformity tests apply to the ranks but are complicated by dependence between cases, such as overlapping trajectories or spatial pooling. The paper runs no tests. Calibration should be paired with sharpness or with proper scoring rules.
- Multivariate ranks relate to Bickel's (1969) multivariate Smirnov two-sample test.

## Caveats

- **Read from the arXiv version (1310.0236); the JCGS version may differ.**
- All evidence is visual: there are histograms but no power analysis or test error rates.
- Every simulation uses Gaussian, ordered (temporal) components. Band depth and average rank need no ordering, but the correlation-shape findings are specific to lag structure.
- A calibrated but uninformative forecast, such as climatology, gives flat histograms under every pre-rank. Sharpness has to be assessed separately.
