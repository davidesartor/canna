# Graphical Test for Discrete Uniformity and Its Applications in Goodness-of-Fit Evaluation and Multiple Sample Comparison

Säilynoja, Bürkner & Vehtari, 2022, Statistics and Computing 32 · [arXiv:2103.10522](https://arxiv.org/abs/2103.10522)

**Evidence: B.** Peer-reviewed (Statistics and Computing, DOI 10.1007/s11222-022-10090-6); 82
citations in 5.6 yr (Semantic Scholar, 2026-10-22).

## What they do

SBC produces ranks or PIT values that should be uniform. Histograms are the usual check, but they depend on the binning. The authors replace them with the empirical CDF of the PIT values plus simultaneous confidence bands. The bands cover the whole ECDF at the chosen level, not each point separately.

For N values, N·F̂(zᵢ) is Binomial(N, zᵢ) under uniformity at each evaluation point zᵢ. The band is set so that the ECDF stays inside it everywhere with probability 1 − α. Two ways to find it:

- **Simulation.** Draw uniform samples, record the smallest pointwise binomial tail probability γ_m in each, and take the α quantile.
- **Optimisation.** Compute the exact coverage with a Markov-chain recursion over the evaluation points, and solve for γ with Brent's method.

They recommend the ECDF-difference plot (ECDF minus the identity), which makes small deviations visible. They also give a multi-sample version for comparing chains, with hypergeometric marginals.

## What they find

- Speed of the optimisation method: about 600 ms versus 10 s for simulation at N = 250, and 10 s versus 75 s at N = 1000.
- The bands hold their nominal coverage within ±1% for N from 50 to 2000.
- Histograms mislead. With N = 250 one sample passes with 20 bins and fails with 50.
- Shapes:
  - A tested posterior that is too narrow gives a ∪-shaped rank histogram.
  - One that is too wide gives a ∩-shaped histogram. In the ECDF-difference plot it shows as a fast rise above zero then a fall below it, crossing in the middle.
- Eight-schools SBC (500 prior draws, 150 thinned draws from a 3000-draw chain): the test rejects. Ranks of τ pile up at small values.
- Power at N = 100 is comparable to the T1, NS, U² and KS tests, with the added benefit of showing where the deviation is.

## Implementation details

- Thin MCMC draws before ranking. Thinning by S / ESS works, using tail or bulk ESS; this matches Talts et al.'s quantile-ESS rule. For an AR(1) chain with φ = 0.95 they thin by 18, and with φ = −0.95 by 7.
- The test needs only the ranks, so it needs only posterior samples and a simulator. No density, no likelihood.
- Code: github.com/teemusailynoja/simultaneous-confidence-bands. Also in `bayesplot` (`ppc_pit_ecdf`) and the `SBC` R package.

## Caveats

- One-dimensional. Each scalar gets its own band. Joint checks come only from the scalars one chooses, and there is no correction across many plots.
- The test checks uniformity, so it inherits SBC's limits: a sampler that returns the prior passes on every parameter.
- The text extracted from the arXiv PDF had garbled digits; numbers above were decoded and cross-checked against the figures, but a typo cannot be ruled out.
