# Effective Bayesian Heteroscedastic Regression with Deep Neural Networks

Immer, Palumbo, Marx & Vogt, 2023, NeurIPS 2023 · [OpenReview:A6EquH0enk](https://openreview.net/forum?id=A6EquH0enk) (no arXiv version found)

**Evidence: B.** Peer-reviewed (NeurIPS 36); 7 citations in 2.9 yr (OpenAlex, 2026-10-25;
Semantic Scholar rate-limited). The count looks low for the venue.

## What they do

Two parts. First, a natural parameterisation of the Gaussian (Sec. 3, Eqs. 3 to 5): the network outputs f ∈ R², with η₁ = μ/σ² = f₁ and η₂ = -1/(2σ²) = -g₊(f₂), g₊ = exp/2 or softplus. The loss gradients become the plain residuals (μ - y) and (σ² - (y² - μ²)), with no 1/σ² weighting on the mean. No stop-gradients.

Second, a linearised Laplace approximation with a generalised Gauss-Newton Hessian. In natural parameters the Hessian is λ₁λ₁ᵀ - λ₂λ₂ᵀ and the GGN is positive definite (the mean-variance Hessian is indefinite, App. A). Layer-wise Gaussian priors have precisions δ_l set by empirical Bayes: maximise the Laplace marginal likelihood during training (Alg. 1). The marginal likelihood also serves as the early-stopping signal. The predictive is a closed-form Gaussian with epistemic variance J_μ Σ J_μᵀ on the mean plus aleatoric σ²(x*).

## What they find

- UCI test LL (Table 1, 8 datasets; checked against the NeurIPS PDF): the most wins and ties go to naive NLL with empirical Bayes and the posterior predictive (5 wins, 6 ties; energy -0.46, yacht -0.05). Natural NLL with EB and the posterior predictive is second (2 wins, 5 ties; boston -2.36 vs -2.88 for naive NLL with grid search, energy -0.71 vs -1.06, yacht -0.51 vs -0.60). The grid-searched baselines (naive NLL, β-NLL 0.5 and 1.0, faithful) get no wins and at most 2 ties. The natural parametrisation's edge is stability without the posterior predictive: the naive EB loss then collapses (boston -6.03, naval -10.9), the natural one does not (0 wins, 3 ties).
- Image regression with label-dependent noise (Table 2, MNIST MLP and FashionMNIST CNN): Natural EB MNIST LL -5.30, DKL 0.45, RMSE 19.1. FMNIST LL -5.34, RMSE 18.1. Faithful on FMNIST gets RMSE 51.8. β-NLL (0.5) gets RMSE 20.7 and 23.5. The homoscedastic baseline gets RMSE 25.6 (MNIST).
- Rotated MNIST linear probe on last-layer features (Fig. 2): homoscedastic 52.2, faithful 51.3, β-NLL (0.5) 71.6, β-NLL (1.0) 69.3, natural Laplace 76.8.
- β-NLL and faithful test NLL vary strongly with prior precision (Fig. 2), which motivates automatic regularisation.
- Faithful's shared encoder struggles when the variance depends on inputs the mean ignores (App. C, Fig. 5).
- Runtime (Table 7): EB cuts training to 9 min from 49 min for natural NLL with grid search.
- CRISPR-Cas13 (Fig. 3): natural EB with the posterior predictive is best. Exact numbers are not in the text.

## Implementation details

- UCI/CRISPR: one hidden layer of 50 units (GeLU/ReLU), Adam, batch 256, 5000 epochs, lr 1e-2 cosine to 1e-5.
- Images: MNIST MLP 3×500, FMNIST CNN, SGD batch 512, 500 epochs, lr 0.1 cosine.
- EB schedule: burn-in 100 epochs, update every 50 epochs, 50 hyper-steps, hyper-lr 0.01 to 0.001 (images: 50 / 25 / 50, lr 0.1). Full Hessian for UCI, KFAC for CRISPR and images.
- Baselines: β-NLL at 0.5 and 1.0, faithful, MC-Dropout, mean-field VI, with grid-searched regularisation.
- No separate variance warm-up. EB starts after a burn-in.
- Code: github.com/aleximmer/heteroscedastic-nn.

## Caveats

- The method is a parameterisation plus Laplace, not a loss weighting. Much of the gain comes from empirical-Bayes regularisation and the posterior predictive.
- Univariate Gaussian targets. The Hessian is for the two natural parameters of one scalar target.
- Models are small (50-unit MLPs, small CNNs). Full Laplace is costly, KFAC is the scalable route. The single-mode Laplace and online EB break the stationarity assumption.
- Epistemic uncertainty in the closed form covers the mean only.
- The natural parameterisation changes the output head. It is not a drop-in loss for an x-prediction network with a fixed anchor and scale.
