# TIC-TAC: A Framework for Improved Covariance Estimation in Deep Heteroscedastic Regression

Shukla, Salzmann & Alahi, 2024, ICML 2024 · [arXiv:2310.18953](https://arxiv.org/abs/2310.18953)

**Evidence: B.** Peer-reviewed (ICML 2024; the arXiv v2 page itself names no venue). OpenAlex (2026-10-25) lists 0
citations on the arXiv record, which probably does not merge the ICML record; the tier rests on the venue.

## What they do

Deep heteroscedastic regression with a full output covariance, trained with Gaussian NLL, plus a label-free metric.
- **Problem:** the covariance has no direct supervision, so a generic map g_Θ(x) can take any value that lowers the NLL without reflecting the prediction's randomness. A too-small predicted variance acts as a larger learning rate (Skafte et al. 2019); poor convergence often comes with large predicted variance (Seitzer et al. 2022).
- **TIC (Taylor Induced Covariance):** perturb the input by x + ε, ε ~ N(0, σ_ε(x)² I). A second-order expansion of the network gives Cov(Ŷ|x) ≈ k₁ J Jᵀ + ℋ + k₃ (Eq. 7). Here J is the input Jacobian, ℋ_ij = k₂ Tr(H_i H_j) uses the input Hessians, and k₃ is a PD matrix for noise independent of the neighbourhood. The cross term vanishes by symmetry (Eq. 4). k₁, k₂, k₃ come from g_Θ(x). Trained jointly with the mean with NLL.
- **TAC (Task Agnostic Correlations):** condition the predicted Gaussian on the true values of all other dimensions, then average |ỹᵢ − yᵢ| (leave-one-out, Alg. 2). It measures correlation quality, which NLL mixes with mean quality.

## What they find

- Synthetic multivariate (Table 1, log-likelihood, dim 4 → 20): NLL −8.2 → −67.2; Faithful −8.7 → −69.0; MSE −10.1 → −66.6; NLL-TIC −7.6 → −39.7.
- UCI, 12 datasets (Table 2): TIC has the best TAC on 10 of 12 (Abalone 1.83 vs 2.54, Electrical 4.89 vs 7.14, White Wine 4.60 vs 6.89). It loses on Naval (0.28 vs 0.20) and Parkinson (2.54 vs 1.68). It is best in likelihood except Naval.
- 2D pose, ViTPose (Table 3, avg TAC / avg LL): NLL-TIC 5.23 / −80.31; Faithful 5.93 / −91.77; NLL 5.80 / −91.61; β-NLL 12.31 / −4.2e3; NLL-Diagonal 12.95 / −204.5; MSE 8.22 / −973.7. Hourglass (Table 4): NLL-TIC 5.64 / −88.6; NLL 6.31; β-NLL 8.06; Faithful 6.27.
- Univariate sinusoids (Fig. 3, qualitative): NLL overestimates the variance. β-NLL and Faithful regularise the variance but compromise the fit. Natural Laplace is unstable when the amplitude varies.

## On NLL failure, mean degradation, β-NLL and Faithful

- **NLL:** prone to sub-optimal convergence, does not overfit, and emphasises the covariance determinant rather than correlations.
- **Mean degradation:** not quantified by the paper. It reports MSE rows and says the MSE-only baseline is skewed toward the mean.
- **β-NLL:** "not the result of a valid distribution", so its optimised values do not translate to a variance. In the tables it has a poor TAC and likelihood on pose (12.31 / −4.2e3 on ViTPose).
- **Faithful (Stirn et al. 2023):** scales the mean gradients by the predicted covariance, so the mean trains with MSE and the covariance with NLL. Immer et al. 2023 are cited for the conflicting residual-correlation assumptions this creates. Faithful is near NLL on pose and close to MSE-level likelihood on the synthetic task.
- The paper does not mention stop-gradients.

## Implementation details

- Jacobian and Hessian by vmap-style parallelisation (Alg. 1). Cost O(n·m³); mitigations proposed are parallelisation and a distilled proxy network.
- Synthetic/UCI: fully connected, batch norm, 100 epochs, reduce-on-plateau; UCI variables standardised to variance ten, 25% of features as inputs, 10 trials.
- Pose: Adam, lr 1e-3, 100 epochs, 5 trials; Stacked Hourglass and ViTPose with soft-argmax over 64×64 heatmaps; 14 joints.
- No train/test split: the authors argue near-singular covariances arise with few samples. Evaluation is the same across methods.

## Caveats

- Regression with an input-space neighbourhood construction. It is not a generative model and has no time axis, sampler or flow.
- The Hessian cost grows steeply with output dimension, which matters for a large output.
- Metrics (TAC, in-sample likelihood) are measured on the training data, not held out. The β-NLL and Faithful runs follow the paper's own setup, not their original tuning, and β is not reported.
- Full covariance only helps when targets are correlated; TIC does not help on Naval, where every sample has low variance.
- Only the HTML up to its end was read, and the "ICML 2024" attribution comes from the task brief.
