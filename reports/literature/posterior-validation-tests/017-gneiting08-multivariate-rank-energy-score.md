# Assessing Probabilistic Forecasts of Multivariate Quantities, with an Application to Ensemble Predictions of Surface Winds

Gneiting, Stanberry, Grimit, Held & Johnson, 2008, TEST 17(2), 211–235 · [DOI:10.1007/s11749-008-0114-x](https://doi.org/10.1007/s11749-008-0114-x)

**Evidence: A.** Peer-reviewed (TEST, with discussion and rejoinder); 326 citations in 18.3 yr (Semantic
Scholar, 2026-10-22).

## What they do

Carries "maximise sharpness subject to calibration" to vector outcomes in low dimension (the case study has d = 2; for large d they suggest low-dimensional functionals). Tools, for an ensemble x₁..x_m and observation x₀:

- **Multivariate rank histogram** (§2.1). (a) Optionally standardise: apply a *principal component transform to the pooled set* {x₀..x_m}. (b) Pre-rank ρ_j = Σ_k 1{x_k ⪯ x_j}, with x ⪯ y iff every component of x is ≤ that of y (Eq. 2); ρ_j counts itself, so it lies in 1..m+1. (c) The rank is drawn uniformly from {s< + 1, …, s< + s=}, where s<, s= count pre-ranks below and equal to ρ₀. Reduces to the Talagrand diagram at d = 1. Fig. 2 example: observation pre-rank 3, rank uniform on {5, 6}.
- **MST rank histogram** (§2.2; Smith 2001). l_j is the length of the MST of the pooled set without x_j; the rank is that of l₀. Fails at d = 1. Underdispersed or biased ensembles fill the *lowest* ranks, overdispersed the highest — a center-outward reading, unlike the ⪯ rank. Footnote 2: distance-based, so it "depends strongly on the units"; the multivariate rank much less.
- **General exchangeability framework** (§2.3). Any F(x_j; x₋ⱼ) symmetric in its set argument gives a uniform rank of z₀ under exchangeability. Which F to prefer has "no obvious answer".
- **Density forecasts** (§2.4). The direct multivariate PIT is not uniform (Genest & Rivest 2001); the stepwise Rosenblatt PIT is, but depends on the variable order. They use **Box's density ordinate transform**, u = 1 − P(p(X) ≤ p(x₀)), which is 1 − χ²_d of the Mahalanobis distance for a Gaussian forecast; randomised if discrete, at a cost in power.
- **Determinant sharpness** (§3, Eq. 3): DS = (det Σ)^{1/(2d)}, Σ the ensemble covariance; needs m > d.
- **Energy score** (§4, Eq. 6): es(P, x) = E‖X − x‖ − ½E‖X − X′‖, proper (Gneiting & Raftery 2007), = CRPS at d = 1. For an ensemble (Eq. 7) the spread term is (1/(2m²))Σᵢⱼ‖xᵢ − xⱼ‖; for a point forecast it is ‖μ − x‖ (Eq. 8). Standardise incommensurable components first.
- **Deviation from uniformity** is quantified by the discrepancy Δ = Σⱼ|fⱼ − 1/(m+1)| (Eq. 1) and read diagnostically. Formal tests "require care in interpretation" (citing Hamill 2001); none is run.

## What they find

- **Simulation** (Fig. 3; observation N(0, I₂), m = 8 from N((d, 0), diag(r, r)), d ∈ {0, 0.5, 1}, r ∈ {0.3, 0.5, 0.8, 1, 2, 5}, 10,000 cases per panel). The multivariate rank gives the familiar shapes: U for r < 1, hump for r > 1, skew for d > 0. Fig. 4 repeats it for the MST rank.
- **Case study** (§5; UW mesoscale ensemble, m = 8, surface wind vectors, 31 Oct 2002–31 Mar 2004, 565 stations with more than 300 cases). ED is bias- and spread-corrected by dressing with past errors; CLI samples the site's own observed winds. Table 2 (ES / ensemble-mean Euclidean error / Δ / DS): UW 2.95 / 3.41 / 0.64 / 0.54; ED 2.32 / 3.29 / 0.04 / 6.95; CLI 2.09 / 3.01 / 0.12 / 6.29. UW is "severely underdispersed", ED and CLI "calibrated"; CLI is sharper than ED and wins on the ES.
- **Density versions** (Table 3, ES / LogS / QS / SphS): UW 2.92 / ∞ / 0.43 / −0.05; ED 2.34 / 4.71 / −0.01 / −0.12; CLI 2.12 / 4.23 / −0.04 / −0.17. All scores order CLI < ED < UW. UW's log score is infinite in some cases. BOT histograms: ED and CLI nearly uniform, UW underdispersed.
- **Calibration does not separate climatology from a skilful forecast** (§5.4). At Olympia (397 observations, Table 4) ED and CLI both have nearly uniform rank histograms (Δ 0.21 and 0.19); ED wins on the ES (1.95 vs 2.27) because it is sharper and gives a better state-dependent point forecast. At West Vancouver (364 observations, Table 5) CLI wins outright (ES 0.83 vs 2.58), attributed to terrain the 12 km model cannot resolve.

## Discussion and rejoinder (TEST 17, 236–264)

- **Uniformity is necessary, not sufficient** (Fuentes): a flat pooled histogram can hide subpopulations; she suggests a Mahalanobis distance for the MST when m > d.
- **Conditional non-uniformity** (Dawid): the rank is uniform marginally, not conditional on the ensemble. The rejoinder links this to conditional exceedance probabilities (Mason et al. 2007), i.e. checking calibration conditional on a forecast-side statistic.
- **Pre-rank choice.** The rejoinder prefers the ⪯ rank because it "minimizes user-dependent choices"; for the MST, the distance choice is critical. Delaunay-hypervolume ranks give infinite volumes for outer members, hence ties, and their distance choice is again critical.
- **BOT catches what orderings miss**: an observation in a low-density region between modes (Box 1980), which ⪯ and MST ranks can place as typical. BOT depends on the parametrisation (Cox objected, Box did not); the authors side with Box.
- **Sharpness** (Jolliffe): DS misleads for near-singular Σ; the rejoinder suggests dimension reduction first. Uniformity tests with more power: Elmore (2005), Jolliffe & Primo (2008). Rejoinder: sharpness means small within-forecast and large between-forecast spread.
- **Energy score.** Winkler & Jose: combining incommensurable dimensions into one Euclidean distance is "particularly problematic"; sensitivity to distance "is inconsistent with" locality; a score needs a suitable baseline distribution. Angulo & Ruiz-Medina ask whether the ES captures interdependence "without joint standardization". Bickel suggests the PIT of the projections ⟨v, X⟩ for a collection of unit vectors v, since "there is nothing sacred about quadrants". The name follows Székely (2003).
- **Dimension**: the rejoinder states the tools "apply readily to higher-dimensional" forecasts, with a four-variate example.

## Implementation details

- ES cost is O(m²·d) for an ensemble. For densities, footnote 4's Monte Carlo estimator, (1/k)Σᵢ‖xᵢ − x‖ − (1/(2(k − 1)))Σᵢ₌₁^{k−1}‖xᵢ − xᵢ₊₁‖ with k = 10,000 draws, uses consecutive pairs only and is O(k·d). That is the form to use with many flow draws.
- Eq. 7 includes the i = j zeros, so its spread term is biased low by (m − 1)/m; the paper does not discuss a "fair" correction.
- Ties among ⪯ pre-ranks are broken at random (step c) or by Czado et al. (2007)'s non-randomised construction.
- The PCA standardisation in step (a) is fit on each case's pooled set, so it changes which pairs are comparable; in the case study it is omitted.

## Caveats

- **The "applies readily" claim did not hold up.** Thorarinsdottir et al. (2016; [008](008-thorarinsdottir16-band-depth-average-rank.md)) find the ⪯ rank nearly flat at d = 15 despite severe dispersion errors, because few vectors are comparable and random tie-breaking flattens the histogram. This is a later finding, not the paper's. Our 8-coordinate × several-source posterior is in that regime.
- The ES's weak sensitivity to dependence is also later ([016](016-scheuerer15-variogram-score.md)).
- Only bivariate Gaussian simulations; no power study, no formal test; calibration is judged by eye and Δ.
- Pooled over cases, CLI is calibrated by construction (it samples the outcomes' own distribution), the multivariate analogue of q = prior passing SBC. Only the ES and sharpness separate it, and at West Vancouver they favour it.
- Full text read (journal PDF, via text extraction), with the seven discussion contributions and the rejoinder.
