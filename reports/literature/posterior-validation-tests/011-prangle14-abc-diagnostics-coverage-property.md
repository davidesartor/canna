# Diagnostic Tools for Approximate Bayesian Computation Using the Coverage Property

Prangle, Blum, Popovic & Sisson, 2014, Aust. N. Z. J. Stat. 56, 309–329 · [arXiv:1301.3166](https://arxiv.org/abs/1301.3166)

**Evidence: B.** Peer-reviewed (ANZJS, DOI 10.1111/anzs.12087); 72 citations in 12.8 yr
(Semantic Scholar, 2026-10-22).

## What they do

An approximate posterior has the coverage property if its credible intervals contain the truth at their nominal rate when θ₀ is drawn from some distribution H and y from the model. Result 3: coverage holds if and only if the p-value p₀ = Q(θ₀ | y) is uniform. This is SBC applied to ABC.

Result 2 is the key negative result: the prior itself satisfies coverage under any H with density h(θ₀) = π(θ₀). So prior-wide coverage cannot detect an uninformative posterior. Their fix is to restrict the test to data near y_obs: draw θ₀ from a truncated prior so that the simulated y₀ falls in a region A around y_obs. Under this truncation the prior no longer passes.

Algorithm 2 makes this cheap:

1. Simulate one bank U of N (θ, y) pairs.
2. Take V, the c simulations whose summaries are closest to y_obs.
3. For each pseudo-observed dataset in V, run ABC with leave-one-out on the same bank, for each tolerance ε.
4. Compute p = (1 + Σⱼ I(θⱼ < θ₀)) / (2 + n) from the n accepted draws.

## What they find

- A χ² statistic on p-values is blind to the symmetry p ↔ 1 − p. They show a clearly non-uniform histogram that χ² passes with p = 0.75. They use the KS statistic instead, and a binned calibration plot.
- For model choice they give three statistics: U (a proportion), V (a Bernoulli log-likelihood) and W.
- Toy model choice, N(0, 1) versus g-and-k (g ~ U(0, 4), g_obs = 0.2, 100 data points, quartile summaries, 2 × 10⁶ simulations):
  - With V drawn from the full prior, coverage holds at both large and small ε. The test says nothing.
  - With V drawn from the truncated prior, the prior fails, and the test separates good from bad tolerances.
  - ε ≤ 0.28 is sufficient.
  - Re-simulating W instead of reusing the bank gives nearly identical results.
- Human demography (Sjödin et al. 2012; 3 models × 9 parameters; 10⁵ simulations per model; 0.5% accepted; ε = 1.36): coverage fails without regression adjustment and holds approximately with it, with minor concerns on N_B and T_dur.
- Default c = 200. c = 100 and c = 500 gave similar results.

## Implementation details

- Needs only posterior samples and a simulator. The bank is reused for both the pseudo-observed datasets and the ABC fits, so the test adds almost no simulation cost.
- The neighbourhood around y_obs is defined on the summary statistics, not the raw data.
- Code: the `abc` R package.

## Caveats

- Summaries that carry no information pass trivially, because the ABC posterior is then the prior restricted to the neighbourhood, which is also what the truncated prior gives. The authors say so.
- Checks are per parameter. Joint failures can be missed.
- It is not proven that only the true posterior satisfies truncated coverage.
- The truncated test checks calibration near y_obs only, and its power depends on how the neighbourhood is chosen.
