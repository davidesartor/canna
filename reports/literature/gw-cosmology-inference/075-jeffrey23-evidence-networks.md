# Evidence Networks: simple losses for fast, amortized, neural Bayesian model comparison

Jeffrey et al., 2023, MLST 2024; arXiv v2 Jan 2024 · [arXiv:2305.11241](https://arxiv.org/abs/2305.11241)

**Evidence: B.** Peer-reviewed (Machine Learning: Science and Technology); 27 citations in 3.4
yr (Semantic Scholar, 2026-10-19).

## What they do

Bayesian model comparison as classification on (x, model label) pairs simulated from each model with parameters drawn from their priors. The parameters are never seen by the network. A bespoke loss makes the network output a monotone function of log K directly, instead of p(M|x) via cross-entropy. Taking a ratio of two estimated posteriors loses precision when K is large.

## What they find

- **Time-series toy:** linear Gaussian with dim θ = dim x = 100 and an analytic K.
  - Setup: 10⁶ training sims, an ensemble of 4 dense networks.
  - log K RMSE ≈ 0.02 over many orders of magnitude.
- **Loss comparison:**
  - Cross-entropy shows a systematic bias at moderately high K.
  - The exponential loss reduces it.
  - l-POP-Exp (α = 2) removes it visibly.
- **vs PolyChord:** with known likelihood, using about 1% of PolyChord's likelihood-evaluation count as training sims, Evidence Networks give lower log K error across dim θ.
- **vs normalising-flow ratio:** at dim x = 20, the flow ratio p(x|M₁)/p(x|M₀) from 10⁶ samples has more than 10× the RMSE of the Evidence Network.
- **DES Y3 cosmic shear C_ℓ, NLA intrinsic alignment vs none:**
  - Setup: 2×10⁴ sims per model, an ensemble of 13 networks.
  - log₁₀K = −0.8 ± 0.3 (jackknife over the ensemble).
  - α = 2 overfitted for some members, so they used α = 1.
- **Rastrigin toy:** an exponentially multimodal posterior, yet the Bayes factor is easy to learn.

## Implementation details

- **Loss:** use the l-POP-Exponential loss with α = 2 by default. Lower α if ensemble members overfit or disagree.
  - α cannot be chosen by validation loss, because changing α changes what the loss means.
  - Choose it by the coverage test instead.
- **Blind coverage test:**
  - Bin validation sims by the predicted p(M₁|x) = sigmoid-like transform of the estimated log K.
  - Compare the fraction of M₁ labels in each bin, using binomial error bars.
  - Rescaled residuals should be N(0,1).
  - This validates the network without ground-truth K.
- **Ensembles:** averaging 4–13 independently initialised networks "significantly improves" accuracy, and the ensemble jackknife gives an error bar.
- **Augmentation:** use data symmetries, e.g. sign flip in the toy.
- **Priors:**
  - The model prior is the class balance, which can be reweighted afterwards.
  - A change of parameter prior can be handled by θ-dependent loss weights, i.e. importance weights.
- **Uncompressed data:** feed x as is. Compression changes the evidence, whereas the network can use raw x.
- **Absolute evidence:** obtainable against an analytic reference model. A two-network trick gives posterior-predictive odds.

## Caveats

- Only binary, nested comparisons are shown.
- Accuracy is limited to the range of K the training set covers. Extreme K values are rare in the training set, so the extremes are learned poorly.
- The result depends on the prior, as Bayes factors always do.
- The DES example uses a simplified systematics model.
