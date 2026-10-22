# Likelihood-free posterior validation takeaways for canna.lisa

What we take from the papers in this folder for testing the flow's posterior from simulations
and its own draws alone, without the likelihood and without an MCMC run per injection, and what
should replace the scorecard's calibration columns. Comparing against MCMC reference posteriors
and the case for width-first metrics are in
[../gw-cosmology-inference/000-takeaways.md](../gw-cosmology-inference/000-takeaways.md) §4.

The setting: an amortised flow q(θ | y, f) over 4 sources × 8 flat coordinates per window,
ordered loudest first. We can draw from q but not evaluate it (log q needs an ODE divergence
integral). The simulator is cheap and gives as many prior-predictive pairs (θ*, y) as we want.
Two failure modes matter:

- **Some parameters come out as the prior.** The phases of z± have been drawn from the prior
  while the MCMC pins them.
- **Loud sources are too wide.** Their widths sit tens of times above the reference, while quiet
  sources are close to it.

The scorecard today reports the f0 width against the monochromatic Fisher bound, a "found"
fraction (width < 1 bin), and f0-only calibration (in68, in95, mean rank). It runs on 10
injections picked at the SNR deciles plus 200 prior draws, with 256 draws each. It relabels the
draws onto the truth with `match_sources`, and it bands sources by their true SNR.

## How much to trust each source

The tiers are those of [the GW folder](../gw-cosmology-inference/000-takeaways.md#how-much-to-trust-each-source):

- **A, established:** ≥100 citations, or peer-reviewed with ≥30 citations a year.
- **B, credible:** peer-reviewed, or a preprint with ≥20 citations.
- **C, unvetted:** a hypothesis.

Semantic Scholar counts were taken on 2026-10-22. Statistics journals cite less than ML venues,
so Bayesian Analysis and JCGS papers often land at B. Six papers on this topic were already
reviewed in the GW folder and are linked from there, not duplicated here:

- Talts+18 SBC ([gw/018](../gw-cosmology-inference/018-talts18-validating-bayesian-inference-algorithms.md) (A))
- Zhao+21 local coverage ([gw/019](../gw-cosmology-inference/019-zhao21-diagnostics-conditional-density-models.md) (B))
- Lemos+23 TARP ([gw/028](../gw-cosmology-inference/028-lemos23-sampling-based-accuracy-testing.md) (B))
- Linhart+23 L-C2ST ([gw/035](../gw-cosmology-inference/035-linhart23-l-c2st.md) (B))
- Hermans+22 expected coverage ([gw/054](../gw-cosmology-inference/054-hermans21-trust-crisis-simulation-based.md) (B))
- Lueckmann+21 benchmark ([gw/053](../gw-cosmology-inference/053-lueckmann21-benchmarking-simulation-based-inference.md) (A))

## 0. The one identity behind every test

Draw (θ*, y) from p(θ) p(y | θ), then θ̃₁…θ̃_M from q(· | y). If q is the posterior, then
θ*, θ̃₁, …, θ̃_M are exchangeable given y. So for any scalar f(θ, y), the rank of f(θ*, y) among
the f(θ̃_m, y) is uniform on {0, …, M}. This holds even on any subset of data space
([001](001-modrak23-sbc-test-quantities-sensitivity.md) (B), Thm 5;
[009](009-cook06-validation-software-posterior-quantiles.md) (A);
[gw/018](../gw-cosmology-inference/018-talts18-validating-bayesian-inference-algorithms.md) (A)).

Every test in this folder checks a consequence of this identity. They differ in three ways:
which f or classifier they use, whether they average over y or localise in y, and whether they
test calibration or score sharpness. Each is a necessary condition. A finite set of passes
never proves q correct ([001](001-modrak23-sbc-test-quantities-sensitivity.md) (B)). Draws from
q are independent given y, so none of the MCMC thinning advice applies
([002](002-yao23-discriminative-calibration.md) (B)).

## 1. Plain SBC and coverage cannot see our main failure

- **q = prior passes every data-independent check. Established, and proven several times.**
  - SBC with any f(θ) passes: Modrák+23 Thm 7 ([001](001-modrak23-sbc-test-quantities-sensitivity.md) (B)); Cook+06 say so explicitly ([009](009-cook06-validation-software-posterior-quantiles.md) (A)).
  - So does ABC coverage under the prior: Prangle+14 Result 2 ([011](011-prangle14-abc-diagnostics-coverage-property.md) (B)).
  - So do expected HPD coverage ([gw/054](../gw-cosmology-inference/054-hermans21-trust-crisis-simulation-based.md) (B)), TARP with prior references ([gw/028](../gw-cosmology-inference/028-lemos23-sampling-based-accuracy-testing.md) (B)) and the law-of-total-variance moment check ([010](010-yu21-law-total-variance-assessment.md) (B)).
  - Geweke's joint test passes too with parameter-only test functions, per Cook+06 ([018](018-geweke04-getting-it-right-joint.md) (A)); its Eq. 2 allows g(θ, y), and one coupling θ and y would catch q = prior (our derivation).
  - Measured head-on: SBC and TARP keep power ≈ 0.05 against q = prior, i.e. the false-positive rate, from 3-D to 100-D ([003](003-chen25-conditional-localization-test.md) (B)).
- **The forecast literature reached the same point earlier.** The climatological forecaster
  has a flat PIT histogram and 51.3%/90.7% coverage at 50%/90%. Only sharpness or a proper
  score separates it from the ideal forecaster: CRPS 0.78 against 0.56
  ([004](004-gneiting07-calibration-sharpness.md) (A)). A flat rank histogram does not mean a
  reliable forecast, because pooling cancels conditional errors
  ([015](015-hamill01-rank-histograms.md) (A); Zhao+21 Thm 1,
  [gw/019](../gw-cosmology-inference/019-zhao21-diagnostics-conditional-density-models.md) (B)).
  Multivariate rank histograms pooled over cases are flat for q = prior as well
  ([017](017-gneiting08-multivariate-rank-energy-score.md) (A),
  [008](008-thorarinsdottir16-band-depth-average-rank.md) (B)).
- **What plain SBC does catch.** It catches a posterior that is too wide on average, as a ∩ in
  the rank histogram, or as a bump in the ECDF-difference plot
  ([gw/018](../gw-cosmology-inference/018-talts18-validating-bayesian-inference-algorithms.md)
  (A), [007](007-sailynoja22-graphical-test-discrete-uniformity.md) (B)). The moment version
  shows the same as Σ_R > Σ_L ([010](010-yu21-law-total-variance-assessment.md) (B)). Our
  too-wide loud sources are a small share of prior draws, though, so pooled over the prior
  their ∩ is diluted. Their width shows only in a stratum (§4).

## 2. Make the test depend on the data

- **Data-dependent test quantities are the established fix.** Ranking f(θ, y) instead of f(θ)
  keeps the guarantee of §0. With the joint log-likelihood as f, q = prior fails "within a
  handful of simulations" while every parameter-only quantity passes
  ([001](001-modrak23-sbc-test-quantities-sensitivity.md) (B)). Products and differences of
  correlated parameters catch a wrong posterior correlation that the marginals pass (same
  paper). Modrák+23's default set:
  - every parameter;
  - the joint log-likelihood;
  - the log prior;
  - a few products or differences of correlated pairs.
- **Likelihood-free ways to make f depend on y.** All of these rank a distance from the draws to
  something computed from y:
  - **TARP with data-dependent references.** Centre the reference point on a cheap estimate
    from y, plus noise. That makes TARP sufficient and catches q = prior
    ([gw/028](../gw-cosmology-inference/028-lemos23-sampling-based-accuracy-testing.md) (B)).
  - **CoLT.** It learns that centre, θ_l(y), as a network. The plain-Euclidean variant (CoLT
    ID) has power 1.000 against q = prior from 3-D to 100-D, where C2ST falls to 0.12. It also
    detects covariance scaled by (1 + α), i.e. too wide
    ([003](003-chen25-conditional-localization-test.md) (B)). That is one paper, NeurIPS 2025,
    synthetic benchmarks with y ≤ 100-D. So treat it as credible but unreplicated.
  - **Neighbours of an observation.** Restrict the truths to simulations whose y lies near a
    given observation ([011](011-prangle14-abc-diagnostics-coverage-property.md) (B),
    [010](010-yu21-law-total-variance-assessment.md) (B)). Posterior SBC does the same through
    a refit on doubled data ([012](012-sailynoja25-posterior-sbc-conditional-data.md) (B)),
    which our fixed-window flow cannot take.
- **The y-dependent part must not come from q itself (our reading).** If the centre or summary
  is q's posterior mean, it is the prior mean when q = prior, and the test is blind again. It
  needs an estimator of θ from y that is independent of q.
- **Simple y-only estimators exist for this problem (our reading, untested).**
  - f0: the periodogram peak bin.
  - The phases of z±: the phase of the data at the peak bin, relative to a template at the
    drawn f0.

  Both are deterministic functions of y, so ranking a draw's distance to them is valid. How
  much power they have is the open question.
- **Using the forward model as the test quantity.** A whitened residual of y against the
  noiseless signal at θ is the log-likelihood up to a constant. It is Modrák+23's strongest
  quantity, and the flow-only rule allows it at evaluation time. It is not likelihood-free,
  though. A fully simulator-only variant ranks the distance from y to a fresh noisy simulation
  at each draw. It stays valid, because independent noise per draw keeps exchangeability (our
  derivation). Its power is likely poor: the noise term in ‖y − y′‖² grows like the number of
  bins, while the signal term is a handful of bins.
- **The posterior mean must be the best point estimate (textbook).** The posterior mean
  minimises E‖θ − T(y)‖² over every estimator T. If any independently trained regressor from
  y beats the mean of q's draws in held-out MSE, q is wrong. For q = prior the flow's MSE is
  the prior variance. The check is cheap and likelihood-free, but blind to width: it tests
  the mean only. No paper in this folder runs it as a posterior test.

## 3. Classifier tests on (θ, y)

- **Discriminative calibration generalises SBC.** Yao & Domke train a classifier to tell the
  true θ from q's draws, given y ([002](002-yao23-discriminative-calibration.md) (B)):
  - Its held-out log predictive density lower-bounds a divergence between p(θ | y) and
    q(θ | y), averaged over y. A within-batch permutation test gives an exact p-value.
  - The multiclass mapping is the most powerful.
  - With y as a feature it can detect q = prior; the rank-only mapping reduces to SBC and
    cannot.
  - On Gaussian bias and covariance-scaling (×0.8–1.2) corruptions, it beats Bonferroni SBC.
    It is "sometimes as good as SBC with a 10 times bigger simulation sample size".
- **The catch is the features.** In their 14-parameter cosmology case with d_y up to 1031,
  they fed log q to the classifier. A black-box MLP on raw (θ, y) "would pass a t-test and not
  even detect miscalculation" ([002](002-yao23-discriminative-calibration.md) (B)). Plain
  C2ST on joint samples drops to power 0.12 against q = prior at 100-D
  ([003](003-chen25-conditional-localization-test.md) (B)).
  - Linhart+23's local version needs the classifier to read x too
    ([gw/035](../gw-cosmology-inference/035-linhart23-l-c2st.md) (B)).
  - Kernel conditional tests (ECMMD, [020](020-chatterjee24-kernel-conditional-two-sample.md)
    (C); PQMass on joint samples, [019](019-lemos24-pqmass-probability-mass-estimation.md) (B);
    SKCE, [021](021-widmann21-calibration-tests-beyond-classification.md) (B)) depend on y-space
    distances or neighbours. Those mean little for a long noisy rfft window.
- **What we take.** Classifier tests are the most general, but they move the difficulty into a
  second network that must read y about as well as the flow. A pass is only as strong as that
  network. Use them after the cheap checks, with y compressed by a y-only feature map, never
  by q's own outputs. The classifier-probability-versus-parameter plot is a useful way to read
  a failure: a U or inverted-U trend means under- or over-confidence in that parameter
  ([002](002-yao23-discriminative-calibration.md) (B)).

## 4. Stratify by the data, never by the truth

- **Uniformity holds on subsets of data space, not of parameter space.** The guarantee of §0
  holds conditionally on any function g(y) ([001](001-modrak23-sbc-test-quantities-sensitivity.md)
  (B), Thm 5), and local tests are built on it
  ([gw/019](../gw-cosmology-inference/019-zhao21-diagnostics-conditional-density-models.md)
  (B), [013](013-dey22-local-amortized-diagnostics-reshaping.md) (B)). It does not hold
  conditionally on the true θ*.
- **Banding by true SNR breaks calibration (our derivation).** Even the exact posterior fails
  in a band cut on the true SNR. In the top band the truth sits in the upper tail of its own
  posterior, so the coverage of a correct q comes out wrong there.
- **The fix is a y-statistic.** Band by the median SNR of q's draws instead. That is a function
  of y only, so the guarantee holds. For the same reason, the 10 eval injections picked at the
  true-SNR deciles are not a valid calibration sample. Only plain prior draws are.
- **Labels must not use the truth either (our derivation).** f must be the same fixed function
  of (θ, y) for the truth and for every draw ([001](001-modrak23-sbc-test-quantities-sensitivity.md)
  (B), [gw/018](../gw-cosmology-inference/018-talts18-validating-bayesian-inference-algorithms.md)
  (A)).
  - `match_sources` breaks this: it permutes each draw's sources toward the truth. That pulls
    the draws toward θ*, so a too-wide posterior can look calibrated.
  - The fix: give the truth and every draw the same truth-independent canonical order. Sorting
    sources by their own SNR is the flow's slot order; sorting by f0 also works.
  - Near-equal sources that swap then make a bimodal per-slot marginal. That is legitimate,
    and a correct q is still calibrated under it.

## 5. Calibration is necessary; score sharpness with a proper score

- **The forecasting paradigm: maximise sharpness subject to calibration**
  ([004](004-gneiting07-calibration-sharpness.md) (A)). For us, calibration (§1–§4) rules out
  wrong posteriors. Sharpness tells a prior-like, calibrated q from a good one.
- **The energy score is the sample-based, joint, strictly proper choice**
  ([005](005-gneiting07-strictly-proper-scoring-rules.md) (A)):

  ES(q, θ*) = E‖X − θ*‖ − ½ E‖X − X′‖, with X, X′ ~ q.

  - Over prior-predictive pairs, its expectation is uniquely minimised by the true posterior
    ([006](006-pacchiardi22-scoring-rule-posterior.md) (B),
    [005](005-gneiting07-strictly-proper-scoring-rules.md) (A)).
  - The gap to the true posterior is half the energy distance between q(· | y) and p(· | y),
    averaged over y ([005](005-gneiting07-strictly-proper-scoring-rules.md) (A)). The mean
    held-out score therefore measures distance to the true posterior without ever seeing it,
    up to an unknown constant.
  - It penalises q = prior. The unconditional interval has nominal coverage but interval score
    8.04, against 4.77 for the true one ([005](005-gneiting07-strictly-proper-scoring-rules.md)
    (A)).
- **The unbiased estimator from M draws** ([006](006-pacchiardi22-scoring-rule-posterior.md)
  (B)):

  Ŝ = (2/M) Σ_j ‖θ̃_j − θ*‖ − 1/(M(M−1)) Σ_{j≠k} ‖θ̃_j − θ̃_k‖

  It is unbiased for any M ≥ 2 at O(M² d), so scores compare across draw counts. The plug-in
  V-statistic is biased and does not. Per coordinate it is the CRPS. Doubling the first term
  makes it twice the usual ES, which does not change rankings.
- **Sensitivity to dependence.** The energy score is weakly sensitive to the dependence
  between coordinates ([016](016-scheuerer15-variogram-score.md) (A); disputed by an
  unreviewed DM-test study cited there). The variogram score with p = 0.5 targets dependence.
  It is proper but not strictly proper, and blind to a common shift, so use it beside the ES,
  not instead (same paper).
- **Limits.**
  - Scores are relative. They rank checkpoints, runs or samplers on the same injections; they
    never say "correct".
  - Compare two models with a paired test on per-injection score differences (Diebold–Mariano,
    cited in [004](004-gneiting07-calibration-sharpness.md) (A) and
    [005](005-gneiting07-strictly-proper-scoring-rules.md) (A)).
  - The ES depends on the metric, so compute it in the whitened flow chart, per source. Pooled
    over the prior, it is dominated by quiet, prior-wide sources. The loud-source width shows
    only in an SNR stratum, cut on the y-statistic of §4. Within such a stratum the score is
    still proper, because the true posterior minimises it at every y.
- **What scores do not replace.** The scorecard's width-to-Fisher ratio gives an absolute
  scale: how far a width is from the attainable one. No likelihood-free quantity in this folder
  gives that. The ES gap is known only up to the unknown score of the true posterior. The
  absolute yardstick stays the narrowband MCMC comparison
  ([gw §4](../gw-cosmology-inference/000-takeaways.md#4-validate-width-not-just-calibration)),
  or the Fisher bound if a likelihood quantity at evaluation is acceptable.

## 6. Joint checks per source, from samples

- **Pre-ranks.** The average-rank and band-depth pre-ranks turn a source's 8 coordinates into
  one rank per injection ([008](008-thorarinsdottir16-band-depth-average-rank.md) (B)). They
  need samples only, cost O(d M log M), and work at d = 15.
  - Average rank reads like a 1-D histogram: ∩ too wide, ∪ too narrow.
  - Band depth reads centre-outward. Too many high ranks means too wide; ∪ or ∩ shapes mean a
    wrong correlation.
  - The ⪯ multivariate rank degenerates at this dimension (same paper on
    [017](017-gneiting08-multivariate-rank-energy-score.md) (A)), so do not use it.
  - Pooled, these are blind to q = prior like any rank (§1). Pair them with §2 and stratify as
    in §4.
- **Read the ranks as ECDFs.** Show every rank set as an ECDF-difference plot with simultaneous
  bands, and summarise it by the γ statistic, not χ² or binned histograms
  ([007](007-sailynoja22-graphical-test-discrete-uniformity.md) (B),
  [001](001-modrak23-sbc-test-quantities-sensitivity.md) (B)). χ² is blind to the p ↔ 1 − p
  mirror ([011](011-prangle14-abc-diagnostics-coverage-property.md) (B)). The bands hold
  coverage within ±1% for 50 to 2000 injections. They are per scalar, with no correction
  across plots, so with 32 coordinates expect a few band crossings by chance.

## 7. Cost

- **Number of injections.** Modrák+23 use N injections of the same order as M draws each
  ([001](001-modrak23-sbc-test-quantities-sensitivity.md) (B)). With N = 1000 and M = 127
  (M + 1 = 128 divides evenly into bins), that is 127k flow draws, 2.4× the scorecard's
  210 × 256. That is one GPU job.
- **Score cost.** The energy score at M = 127 and d = 8 costs nothing next to the ODE.
- **Classifier and CoLT cost.** These need 10⁵–10⁶ (θ, y) examples. That is the same draws
  plus a few GPU-minutes of training ([002](002-yao23-discriminative-calibration.md) (B):
  about 1M examples in about 2 CPU-hours). The permutation null reuses the one trained
  classifier.

## What we take: a validation pass instead of the scorecard

In order of cost and evidence. Each step is a recommendation, not a decision:

1. **Prior-predictive injections only.**
   - Draw N ≈ 1000 (θ*, f, noise) from the prior, with no SNR selection, and M = 127 draws
     each.
   - Give the truth and every draw the same canonical source order (§4), never
     `match_sources`.
2. **Per slot × coordinate SBC.**
   - Run it for the 4 × 8 coordinates, plus products and differences of the known correlated
     pairs (f0–Mc; Re/Im within z±).
   - Show ECDF-difference bands and γ (§6). All coordinates are flat now, so the ranks are
     well defined without circular cuts.
3. **Per-source joint ranks.** Average rank and band depth (§6).
4. **At least one data-dependent quantity, to catch q = prior** (§2). The candidates, from
   strongest to weakest:
   - a residual against the noiseless template (strongest; a likelihood quantity, allowed at
     evaluation);
   - distance to a y-only estimate (periodogram peak f0, data phase at the peak), TARP-style;
   - CoLT ID once the cheap ones pass.
5. **Stratify by the y-statistic.** Cut every rank set and score into bands of the draws'
   median SNR, never the true SNR (§4).
6. **Energy score and CRPS.**
   - Per source and per coordinate, unbiased estimator, per SNR stratum (§5).
   - Report them for every checkpoint, so runs can be ranked with paired tests on the same
     injections.
   - Add the variogram score if correlations matter.
7. **Keep an absolute width yardstick** beside these: the narrowband MCMC, or the Fisher ratio
   if a likelihood quantity is accepted (§5). The likelihood-free tests replace the
   scorecard's calibration columns and its "found" fraction. They do not replace an absolute
   width scale.
8. **Later, if 1–6 pass while MCMC still shows gaps:**
   - discriminative calibration (multiclass, y through a y-only feature map, §3);
   - the posterior-mean MSE against an independent regressor (§2).

## Choices resting only on C evidence or our own derivation

- **Stratify by a y-statistic, not by true SNR; label without the truth (§4).** This follows
  from Thm 5 of [001](001-modrak23-sbc-test-quantities-sensitivity.md) (B) and the definition
  of SBC, but no paper shows the failure numerically. Cheapest test: rerun the scorecard's
  in68/in95 on Fisher draws, which are calibrated by construction for loud sources. Do it
  banded by true SNR, then by draw-median SNR, with and without `match_sources`.
- **The y-only estimators of §2 have enough power against q = prior.** Untested. Cheapest
  test: rank the distance to the periodogram-peak phase among draws whose phases are replaced
  by prior draws. That is a constructed q = prior.
- **The fresh-simulation distance is too weak to use (§2).** Our arithmetic, not measured.
- **No recommendation rests on ECMMD ([020](020-chatterjee24-kernel-conditional-two-sample.md)
  (C)), the only C paper here.** CoLT's q = prior result
  ([003](003-chen25-conditional-localization-test.md) (B)) is a single unreplicated paper.
