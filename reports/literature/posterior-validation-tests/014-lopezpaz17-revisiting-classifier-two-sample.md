# Revisiting Classifier Two-Sample Tests

Lopez-Paz & Oquab, 2017, ICLR 2017 · [arXiv:1610.06545](https://arxiv.org/abs/1610.06545)

**Evidence: A.** Peer-reviewed (International Conference on Learning Representations); 596
citations in 6.0 yr (Semantic Scholar, 2026-10-22).

## What they do

This is the reference paper for the classifier two-sample test (C2ST). Pool n samples from P (label 0) and n from Q (label 1). Shuffle and split into train and test sets. Train any binary classifier on the train set. The statistic is the test accuracy t̂. Under H₀: P = Q, n_te·t̂ ~ Binomial(n_te, ½), approximately N(½, 1/(4n_te)). Reject when the accuracy is too far above chance.

The paper is unconditional: it compares two sample sets. Posterior-validation methods such as L-C2ST and the SBI-benchmark C2ST build on it by running it on joint (θ, y) samples.

## What they find

- **Power** (Thm. 1). If the classifier has accuracy ½ + ε under H₁, the approximate power is Φ((ε√n_te − Φ⁻¹(1−α)/2) / √(¼ − ε²)). Power rises with test-set size and with effect size, so the train/test split is a trade-off. Fitting the classifier needs training data, and power needs test data.
- **Type-I control.** At α = 0.05 with N(0,1) vs N(0,1), both C2ST variants stay at or below the level for n ∈ {25 … 10,000} over 100 repetitions.
- **Gaussian vs Student-t** (standardised, 1-D, n = 2000 or ν = 3): the Kuiper test is best, followed closely by C2ST-NN and ME. Wilcoxon fails because the rank means coincide.
- **Independence via permutation** (y = cos(δx) + noise): C2ST-NN is the most sample-efficient, and C2ST-KNN is the most robust to high-frequency δ.
- **NIPS paper categories** (α = 0.01, 500 trials): C2ST-NN reaches power 1.00 in every pair. Its type-I error is 0.002 (Bayes vs Bayes, n_te = 215), against 0.004–0.022 for MMD/ME/SCF.
- **Facial expressions** (α = 0.01): power 0.997, against 1.00 for SCF-full and quadratic MMD. Type-I error 0.002.
- **GAN evaluation** (DCGANs, ResNet-34 features, LSUN): accuracies are far from ½ even for good GANs. In one Table 3 row, MMD = 0.012 against KNN 0.798 and NN 0.964. The accuracy gives a scale where linear MMD is near 0.
- **Causal discovery** (Tübingen pairs): CGAN-C2ST gets 73% (KNN) and the 100-CGAN ensemble 82%.

## Implementation details

- **Classifiers.** C2ST-NN uses one hidden layer of 20 ReLU units, trained for 100 epochs with Adam. C2ST-KNN uses k = ⌊√n_tr⌋. Making either classifier more flexible did not improve results.
- **Null.** All experiments use the Gaussian approximation N(½, 1/(4n_te)) at α = 0.05, or α = 0.01 for the text and face data.
- **Split trade-off.** Simple classifiers need less training data, which leaves a larger test set. Flexible classifiers find more structure but use up test data. Model selection, such as cross-validation, may be run on D_tr only.
- **Interpretation.** The per-sample probabilities f(z) show where P and Q differ. Learned features, such as first-layer filters or GLM weights, show which inputs drive the difference.
- **Loss.** The 0-1 loss gives high-variance Bernoulli terms (Remark 2). The argument extends to real-valued losses. Their variance then enters the power formula and can be tuned to maximise power.
- **Code.** github.com/lopezpaz/classifier_tests.

## Caveats

- Unconditional. To validate q(θ|y) it must be run on joint samples (θ, y) vs (θ̃, y), and then its power depends on how well the classifier models the dependence on y.
- The accuracy statistic is coarse, with high variance. Later work (L-C2ST, Yao & Domke) uses probability-based statistics for this reason.
- The asymptotic theory assumes fixed dimension d with n → ∞. The high-dimensional case n/d → c is left open.
- No recommended train/test ratio is given.
- Not a posterior-validation paper. There is no SBI experiment.
