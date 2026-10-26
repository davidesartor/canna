# Optimal Training of Mean Variance Estimation Neural Networks

Sluijterman, Cator & Heskes, 2023, Neurocomputing (arXiv v2 Aug 2023) · [arXiv:2302.08875](https://arxiv.org/abs/2302.08875)

**Evidence: B.** Peer-reviewed (Neurocomputing); 58 citations in 3.7 yr (Semantic Scholar,
2026-10-25).

## What they do

They study how to train mean-variance estimation (MVE) networks under the Gaussian NLL ½ log σ² + ½ (y - μ)²/σ². The mean and variance sub-networks share only the input layer. The variance uses an exp output.

Three training strategies (Sec. 4.1):
1. No warm-up: joint training from the start.
2. Warm-up: train the mean alone with a constant variance, then train both jointly.
3. Warm-up and fixed mean: after warm-up, train the variance only.

The stated contribution is separate L2 constants for the mean and variance sub-networks, compared with one shared constant. A linear-model argument (App. B) gives different optimal ridge constants for the two.

## What they find

- Sine toy, σ = 0.01 (Sec. 2, Fig. 3): without warm-up the NLL network fails to fit the mean, where plain MSE succeeds. Warm-up fixes the mean (Fig. 5). Separate regularisation, with stronger variance regularisation, also fixes it without warm-up. Equal regularisation cannot (Fig. 7).
- Quadratic heteroscedastic toy (Fig. 6): variance regularisation 0.1 gives no learned variance. 0.04 learns both. 0.01 inflates the variance on one side. 0 leaves the mean unlearned.
- UCI, outer 10-fold CV, test RMSE (Table 2):
  - Yacht RMSE: no warm-up with equal reg. 8.87, no warm-up with separate reg. 3.31, warm-up with equal reg. 1.00.
  - Energy RMSE: 2.20 (no warm-up, equal), 0.468 (no warm-up, separate), 0.850 (warm-up, equal), 0.507 (warm-up, separate).
- UCI log-likelihood (Table 1): Energy -1.25 (no warm-up, equal) vs -0.685 (separate). Kin8nm 1.27 vs 1.28. Boston Housing is identical across settings.
- After warm-up with separate regularisation, fixing the mean or training jointly makes little difference (Sec. 4.4). Separate regularisation is never significantly worse than equal.

## Implementation details

- **Architecture:** each sub-network has 2 hidden layers of 40 and 20 ELU units. Mean output linear. Variance output exp with a floor of 1e-6.
- **Warm-up init:** variance output bias set to 1, so the variance starts roughly constant and every point weighs the same.
- **Optimiser:** Adam, gradient clipping at 5, batch 32, 1000 epochs per stage. Inputs and targets standardised.
- **Regularisation:** inner 10-fold CV picks each constant from {1e-5, ..., 0.1} by held-out LL (5-fold for Kin8nm and Protein). Fallback if no search is possible: regularise the variance about 10× more than the mean (Sec. 5.1).
- Recommendations (Sec. 5.1): warm-up, gradient clipping, separate regularisation.

## Caveats

- Evidence is UCI regression plus toys, with one architecture and L2 only. Folds are few (5 to 10), and many differences are not significant. The authors call the pattern clear but not conclusive.
- Theory is a linear model with asymptotic MLE arguments.
- Warm-up is a hard two-stage schedule on a mean-only loss. Nothing on stop-gradients or per-component variance outputs.
- Mean and variance sub-networks are separate here, so there is no shared trunk for the variance gradient to corrupt.
