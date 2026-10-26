# Faithful Heteroscedastic Regression with Neural Networks

Stirn, Wessels, Schertzer, Pereira, Sanjana & Knowles, 2023, AISTATS 2023 · [arXiv:2212.09184](https://arxiv.org/abs/2212.09184)

**Evidence: B.** Peer-reviewed (AISTATS); 8 citations on the OpenAlex arXiv record in 3.9 yr
(OpenAlex, 2026-10-25; Semantic Scholar rate-limited). The count likely understates the
published version.

## What they do

Define a heteroscedastic network as faithful if its mean error is no worse than the mean-only baseline, meaning the subgraph feeding only the mean head (Defs. 1 and 2). The test is a one-sided paired t-test on squared errors, α = 0.05.

Architecture: shared trunk f_trunk, mean head f_μ, variance head f_Σ (diagonal, softplus). Two changes to the Gaussian NLL (Eq. 5):
1. Newton-style mean step: scale the mean gradient by Σ(x), so Σ·Σ⁻¹ cancels and the mean gradient becomes the SSE gradient.
2. Stop-gradient: the variance head's gradient does not reach the shared trunk.

Theorem 1: with the same seed, the mean and trunk updates are identical to an SSE-trained mean-only model, so the mean fit is guaranteed not to degrade. The variance head still learns on top of the frozen-to-it trunk features.

## What they find

- Synthetic y = x sin x + noise with two isolated points at x = 0.5 and 9.5 (Fig. 2): conventional NLL inflates the variance at x = 9.5 and fails to converge there. The stop-gradient alone (Proposal 2) fails the same way. β-NLL (0.5) is partial. β-NLL (1.0), the mean-gradient scaling alone (Proposal 1) and the full method converge.
- UCI, 10-fold CV (Table 1). Win/tie counts RMSE / ECE / LL: Conventional 1/0/0, β-NLL 0.5: 7/2/3, β-NLL 1.0: 6/1/3, Proposal 1 only 0/0/0, Proposal 2 only 2/1/2, Faithful 11/8/11.
- Example rows (RMSE conventional vs faithful): naval 0.533 vs 0.0247, yacht 0.657 vs 0.0226, boston 0.374 vs 0.304. Faithful equals the unit-variance baseline's RMSE on every dataset.
- LL: naval 3.23 conventional vs 6.63 faithful, yacht -0.769 vs 1.33. Boston is worse: -17.4 faithful vs -2.04 conventional.
- VAE MNIST and Fashion-MNIST (Table 2): faithful RMSE equals the unit-variance model (mnist clean 0.759). LL 119 vs -946, ECE 3.2e-5 vs 9.2e-3. Every other method is unfaithful there.
- CRISPR-Cas13 (Table 3): faithful 3/5/6 wins/ties on RMSE/ECE/LL. β-NLL 0.5 has 6/1/0.
- Ensembles, MC Dropout, Student likelihood (Tables 4 to 6): faithful wins/ties 11/11/9 for ensembles, 9/9/7 for Student.

## Implementation details

- Loss: Σ [ |y - μ|₂/2 - log N(y; ⌊μ⌋, Σ(⌊f_trunk⌋)) ]. The squared error trains mean and trunk. The log-likelihood trains only the variance head, with μ and the trunk features stopped.
- Adam, lr 1e-3 everywhere. UCI: full batch, up to 60k epochs, early stopping on validation RMSE with patience 100, restore the best-RMSE weights, trunk of two 50-unit ELU layers, targets standardised.
- VAE and CRISPR: batch 2048, up to 2k epochs, conv trunks.
- Diagonal covariance, std via softplus. β-NLL compared at β = 0.5 and 1.0.
- No warm-up.

## Caveats

- The mean can never beat the mean-only baseline, so no joint-learning gain for the mean. β-NLL occasionally beats it. Faithful is not always lowest RMSE (Table 3).
- The variance sees a trunk shaped only by the mean loss. Mean and variance relevant features that differ are not learned. Immer et al. (Appendix C) show the shared encoder struggles when variance depends on inputs the mean ignores.
- Theorem 1 holds with identical seeds and updates. Only Gaussian and Student families, diagonal covariance, and no GP comparison.
- Tables 4 and 5 hold identical numbers, a likely transcription slip.
- Aleatoric decomposition needs paired clean and noisy measurements.
