# Towards Instance-Wise Calibration: Local Amortized Diagnostics and Reshaping of Conditional Densities (LADaR)

Dey, Zhao, Andrews, Newman, Izbicki & Lee, 2022 (v8 2025), Machine Learning: Science and Technology · [arXiv:2205.14568](https://arxiv.org/abs/2205.14568)

**Evidence: B.** Peer-reviewed (Machine Learning: Science and Technology, DOI 10.1088/2632-2153/ae1f05); 18
citations in 4.4 yr (Semantic Scholar, 2026-10-22). Earlier versions were titled "Cal-PIT".

## What they do

The method checks calibration of a conditional density estimate f̂(y|x) locally, at every x, instead of averaged over x. In this paper y is the scalar target and x the features (in SBI terms: y is the parameter, x the data).

The quantity they estimate is the PIT-CDF, r_f̂(γ; x) = P(PIT(Y; x) ≤ γ | x), where PIT(Y; x) = F̂(Y|x). If f̂ is correct at x, then r_f̂(γ; x) = γ for every γ.

- **Augment.** For each calibration pair (X_i, Y_i), draw K values γ_ij ~ U(0, 1). Set W_ij = 1(PIT(Y_i; X_i) ≤ γ_ij).
- **Regress.** Fit a regression of W on (x, γ). The fit estimates r_f̂(γ; x) for every x and γ at once, so the diagnostic is amortized: no retraining for new x.
- **Diagnose.** At any x, plot r̂(γ; x) against γ as a local P-P plot. Summarise it as the local discrepancy score LDS D(x) = (1/|G|) Σ_{γ∈G} (r̂(γ; x) − γ)² over a grid G of γ values.
- **Reshape.** The recalibrated CDF is F̃(y|x) = r̂(F̂(y|x); x). Its density follows by numerical differentiation on a y grid. They prove MSE and conditional-coverage guarantees for it.

The paper motivates local checks with the blind-model problem. In their words, "PIT can be optimal even when the model ignores x entirely". Models of the form F(y|g(x)) can pass marginal tests while discarding features (citing Zhao et al. 2021, Thm. 1).

## What they find

- **Photometric redshift** (galaxy photometry → redshift). Initial model: GPz Gaussian mixed with the marginal f(y). CDE loss (lower is better): GPz −9.93, Cal-PIT −10.80. Other methods: FlexZBoost −10.60, CMNN −10.43, TPZ −9.55.
- **Synthetic conditional coverage** (Appendix B). Against Reg-split, CQR and DCP for 90% prediction sets, all methods improve with n. Only Cal-PIT "consistently attains the nominal 90% coverage across the feature space for n ≥ 2000". Bimodal regions get disjoint HPD sets.
- **Tropical cyclone** application (Section 4.2): the initial model is a Gaussian fit. The local P-P plots show where and how it is miscalibrated across feature space.
- The P-P plot shape tells the failure type at each x: bias, or over- or under-dispersion.

## Implementation details

- **Regressor.** Monotonic neural networks (Wehenkel & Louppe 2019), monotone in γ, trained with binary cross-entropy. The paper says BCE "enforces P(PIT(Y;X) ≤ γ|x) to be well estimated". Any regression method works in principle.
- **K.** The number of γ draws per calibration point is a hyperparameter. I did not find a value stated in the text I read.
- **Support.** The initial f̂ must have support over the whole range of y at every x (Remark 1). Their photo-z model adds the marginal f(y) to the Gaussian to widen it.
- **Multivariate y.** Handled by applying the method to autoregressive components (conditional CDFs one at a time), or by the multivariate approach of Linhart et al. 2022.
- **Needs.** A CDF value F̂(Y|x) per calibration pair. With only samples from f̂, the PIT can be taken as the fraction of samples below Y. This is my reading, not a claim of the paper.
- **Code.** github.com/lee-group-cmu/Cal-PIT.

## Caveats

- Scalar targets are the main case. For a joint target, PIT per coordinate or per autoregressive factor checks only those projections.
- The LDS is a diagnostic. I found no formal hypothesis test with a calibrated p-value in v8.
- The regression must learn a function of the full x. Its quality for long raw inputs is not tested here; the applications use tabular photometry or engineered features.
- A blind model (f̂ ignores x) is detectable only if the regression learns how the PIT depends on x. This is my reading, not a claim of the paper.
- I read the arXiv v8 text, not the published version or the supplement.
