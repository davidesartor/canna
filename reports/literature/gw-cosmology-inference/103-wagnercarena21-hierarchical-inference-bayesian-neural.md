# Hierarchical Inference With Bayesian Neural Networks: An Application to Strong Gravitational Lensing

Wagner-Carena et al., 2021, ApJ 909, 187 · [arXiv:2010.13787](https://arxiv.org/abs/2010.13787)

**Evidence: B.** Peer-reviewed (The Astrophysical Journal); 45 citations in 6.0 yr (Semantic
Scholar, 2026-10-19).

## What they do

The training-set distribution acts as an implicit "interim prior" Ω_int on any amortized posterior. The paper deconvolves it hierarchically:
- Population: p(Ω|{d}) ∝ p(Ω) ∏_k (1/N) Σ_{ξ∼q(ξ|d_k,Ω_int)} p(ξ|Ω)/p(ξ|Ω_int), sampled with emcee.
- Per lens: each posterior is importance-reweighted by the same ratio, averaged over Ω draws.
The posterior estimator is an MC-dropout BNN (AlexNet, 64×64 HST-like lensed images). Its output is one of: a diagonal Gaussian, a full Gaussian (log-Cholesky precision), or a 2-component GMM. It predicts 8 PEMD + shear parameters, marginalised over the source.

## What they find

- Training set: 400k images, with noise redrawn on the fly. Training: Adam lr 1e-5, batch 512, 400 epochs, 16–24 h per model on a P100, keeping the weights with the best validation loss.
- Calibration ablation, on a multivariate HPD-like q-q plot:
  - The diagonal model needs 30% dropout to avoid over-confidence. At 10% dropout, 97% probability volume covers just over 80% of truths.
  - Full and GMM models are near-perfect at 0.1% dropout. At 0% they calibrate as well but have notably worse MAE (dropout acts as a regulariser).
  - Diagonal plus large dropout reproduces the full model's total covariance through the epistemic term.
- Hierarchical tests, 1024 lenses each:
  - Centered-narrow sky: the hierarchy fixes the under-confidence from the interim prior and recovers hyperparameters.
  - Shifted-narrow sky: ±1σ_train in each of 8 dims, so only 0.006% of training points fall in the test region. The bias is reduced but the 95% CI misses the truth; the error in the γ_lens mean is about 8% of the shift.
  - Empirical sky with covariances: recovered, except q_lens near 1, which is under-sampled by the Gaussian-in-e1/e2 prior.
- Constraining power scales roughly as √N_lens (64→1024).
- Dropout sweep: an under-confident (0.5–1% dropout) GMM gives systematically underestimated population scatter, despite slightly better MAE.

## Implementation details

- Map network outputs bijectively to valid posterior parameters: log-Cholesky precision, sigmoid mixture weight.
- Standardise targets; normalise each image to unit std.
- Use e1/e2 Cartesian ellipticity and shear coordinates to avoid angle wraparound.
- The training prior must be broader than any plausible test population: reweighting divides by p(ξ|Ω_int).
- Pick models by calibration first, MAE second.

## Caveats

- Same model family in training and test (no simulator misspecification).
- Gaussian/GMM posteriors only.
- The calibration metric uses ellipses centred on the posterior mean, and the paper itself admits it is not unique.
- The forward-model (emcee) posterior is tighter than every BNN: amortization costs precision.
