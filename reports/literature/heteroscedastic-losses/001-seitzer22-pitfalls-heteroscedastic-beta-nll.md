# On the Pitfalls of Heteroscedastic Uncertainty Estimation with Probabilistic Neural Networks

Seitzer, Tavakoli, Antic & Martius, 2022, ICLR 2022 · [arXiv:2203.09168](https://arxiv.org/abs/2203.09168)

**Evidence: A.** Peer-reviewed (International Conference on Learning Representations); 171
citations in 4.6 yr (Semantic Scholar, 2026-10-25).

## What they do

A Gaussian NLL network outputs μ(x) and σ²(x). The NLL gradient on the mean is (μ - y)/σ², so points the model currently fits badly and assigns large σ² get a small learning rate and stop contributing. The mean stays poor while σ² grows to explain it. They call this the "rich get richer" effect, and the 1/σ² weighting is the cause.

Fix, β-NLL (Sec. 4.1, Eq. 7): L = E[ ⌊σ̂^{2β}⌋ · (½ log σ̂² + (y - μ̂)² / (2σ̂²)) ], with ⌊·⌋ a stop-gradient on the weight.
- Mean gradient: (μ̂ - y) / σ̂^{2-2β}. β = 0 is plain NLL. β = 1 gives the MSE gradient on the mean. β = 0.5 weights points by 1/σ̂.
- Variance gradient: (σ̂² - (y - μ̂)²) / (2σ̂^{4-2β}).
- The weight acts as an input-dependent learning rate. The loss value is not a quality measure, so monitor NLL and RMSE.

## What they find

- Sinusoid, σ = 0.01 (Sec. 5.1, Fig. 1, Fig. S2): NLL stays at a poor mean fit after 10^7 updates. MSE reaches the optimum in 10^5. No NLL run gets below RMSE 0.10 (optimum 0.01). With β = 0.5 and 1 the fit is good across learning rates and architectures (Fig. 7).
- UCI, 12 datasets (Table 1, S1, S2). Number of datasets tied with the best on RMSE: MSE 12, β = 1: 9, β = 0.75: 8, β = 0.5: 7, β = 0.25: 6, β = 0: 5. On energy RMSE is 2.25 (β = 0), 1.12 (β = 0.5), 0.92 (MSE). LL ties with best: β = 0.5 and 0.75 have 5, NLL 3, β = 1 only 2. Student-t has 10 LL ties.
- Dynamics models (Table 2): 1D-Slide RMSE 0.0192 (β = 0), 0.0064 (β = 0.5), 0.0068 (MSE). Fetch RMSE 0.00163, 0.00096, 0.00103. Fetch LL 18.72 (β = 0), 24.68 (β = 0.5), 21.32 (β = 1). Moment matching diverges on 1D-Slide LL.
- VAE (Table 3): MNIST RMSE 0.237 (β = 0), 0.151 (β = 0.5), 0.152 (β = 1). LL 2116, 2220, 1706. Student-t reaches LL 4291 but RMSE 0.273.
- Depth, NYUv2 (Table 3, S4): RMSE 0.3854 (β = 0), 0.3789 (β = 0.5), 0.3776 (MSE). LL is worse for β > 0 (-7.50 at β = 0.5 against -4.52 at β = 0).
- β-NLL is stable across a 96-configuration grid (Fig. 9), where plain NLL is sensitive.

## Implementation details

- **Stop-gradient:** on the weight σ̂^{2β} only. The log σ̂² and residual terms still pass gradient to both heads.
- **β sweep:** {0, 0.25, 0.5, 0.75, 1}. They suggest 0.5 as a starting point (Sec. 6). Separate β for mean and variance is mentioned and not tested (App. A).
- **Variance parameterisation:** softplus + 1e-8, clamped at 1000. For depth: min variance 1e-6, max 10.
- **Optimiser:** Adam (0.9, 0.999). UCI: one 50-unit ReLU layer, batch 256, lr grid 1e-4 to 7e-3, up to 20k updates with patience 50. Sinusoid: 2×128 tanh, lr 5e-4, batch 100. VAE: lr 3e-4, batch 256. Depth: AdaBins backbone, 25 epochs, batch 16.
- No variance warm-up phase. No learning-rate warm-up either.
- Code: github.com/martius-lab/beta-nll.

## Caveats

- Per-sample scalar weights with univariate σ. No multivariate, per-component or conditional-generative analysis.
- LL goes down as β goes up (depth, VAE at β = 1, Fetch at β = 1). Student-t, xVAMP and VBEM often have higher LL than β-NLL (Tables 1, 2).
- Best β depends on the task (Sec. 4.2). Calibration is only shown inside the training range.
- No comparison of stopping the variance gradient into the shared trunk. The stop-gradient is on the weight only.
- Variance needs a floor or training can go unstable (App. B.1).
