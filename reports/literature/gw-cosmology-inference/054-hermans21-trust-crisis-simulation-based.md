# A Trust Crisis in Simulation-Based Inference? Your Posterior Approximations can be Unfaithful

Hermans et al., 2021/2022, TMLR · [arXiv:2110.06581](https://arxiv.org/abs/2110.06581)

**Evidence: B.** Peer-reviewed; 77 citations in 5.0 yr (Semantic Scholar, 2026-10-19).

## What they do

Defines the expected coverage probability of the 1−α HPD region of p̂, E_{p(θ,x)}[1(θ* ∈ HPD_{p̂(·|x)}(1−α))], and calls an estimator "conservative" if coverage ≥ nominal at all α. It audits a broad set of SBI methods with this metric and argues that conservativeness, not just accuracy, should be the target for scientific use.

## What they find

- Setup:
  - 7 benchmarks, all with ≤ 3 parameters: SLCP (marginal), Weinberg, Spatial SIR, M/G/1, Lotka-Volterra, Stellar Streams, GW 2 masses.
  - Methods: NPE, NRE, SNPE-C, SNL, SNRE, rejection ABC, SMC-ABC.
  - Simulation budgets 2^10–2^17. Each estimator is trained 5 times for 100 epochs. Coverage is computed from ≥ 5000 test pairs, or 300 observables for sequential methods.
- Every method is overconfident in at least one setting, worst at small budgets and for sequential methods, where the proposal feedback loop compounds errors.
- Ensembles: deep ensembles of 5 consistently raise coverage, and coverage grows with ensemble size (App. E). Bagging does not reliably help.
- Cost: about 200 CPU-days for the amortised and 2800 for the non-amortised experiments. Grid-HPD coverage cost grows exponentially with dim θ.
- All results are figures only; there are no tables.

## Implementation details

- Compute HPD coverage via the density rank: the fraction of samples with p̂ > p̂(θ*).
- Report the coverage curve over α.
- Use ensembles (mixture of members) as a cheap conservativeness fix.
- Post-hoc recalibration by remapping credibility levels is suggested.
- Because coverage alone is satisfied by the prior, also report an information or width metric (expected log posterior density or information gain).

## Caveats

- Low-dimensional benchmarks only.
- Expected (data-averaged) coverage, not conditional coverage.
- HPD needs a density, or a grid in low dimension.
- Advocates conservativeness, which does not help when posteriors are already too wide.
