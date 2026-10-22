# Online stochastic gradient descent on non-convex losses from high-dimensional inference

Ben Arous et al., 2020, JMLR 2021 · [arXiv:2003.10409](https://arxiv.org/abs/2003.10409)

**Evidence: A.** Peer-reviewed (Journal of machine learning research); 155 citations in 6.6 yr
(Semantic Scholar, 2026-10-19).

## What they do

The setting is rank-one estimation on S^{N-1}, where the population loss is φ(m) and m = x·θ is the correlation with the truth. The information exponent k is the order of the first non-zero derivative of φ at the equator m=0. Online spherical SGD (step δ/N) needs a number of samples set by k, and almost all of them go to the search phase: getting from m ~ N^{-1/2} to m = Θ(1).

## What they find

- The thresholds are M = α N with α_c = 1 (k=1), log N (k=2), N^{k-2} (k≥3).
- Thm 1.3 (sufficient): α ≫ α_c, with extra log N or (log N)² factors and a step window α^{-1} ≪ δ ≪ α^{-1/2}.
- Thm 1.4 (refutation): for α ≪ α_c, sup_t |m_t| → 0.
- Thm 1.5 (for k ≥ 2): from m = η to 1-η takes only O(N) samples. The descent fraction is O(1/α_c) → 0, so "essentially all data is used in the search phase."
- Mechanism: the population drift is m_{t} ≈ m_{t-1} + (δ/N) c m^{k-1}. The noise martingale is ~δ√T/N, which forces αδ² = O(1). A larger step does not help, because noise then dominates the weak drift.
- Examples, with k given by Hermite coefficients:
  - f(x)=x and f(x)=x³ have k=1, but x³-3x has k=3.
  - Phase retrieval x², |x| has k=2.
  - Two-component Gaussian mixture: k=1 if the weights are unequal and k=2 if p=1/2. This is the "easy-to-critical transition", and symmetric mixtures sit exactly on the critical boundary.
- Figs 2.1–2.3 (N=3000, α=30,000): with the same data, x³ recovers immediately, x² after a plateau, and x³-3x not at all within budget.
- Fig 2.2: from a warm start m0=0.5, all activations converge in O(N) samples, nearly independent of k.

## Implementation details

Do not fight a high exponent with a larger learning rate (that breaks αδ²=O(1)). Fix it with a warm start or initialisation that already has m = Θ(1), or with a loss or representation whose first Hermite coefficient is non-zero. Breaking a ± symmetry (p≠1/2) drops k from 2 to 1.

## Caveats

The results cover single-index models on the sphere with online (fresh-sample) SGD and an asymptotic N. There is no deep net, no Adam or Muon, and no multi-index structure.
