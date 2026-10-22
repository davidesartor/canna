# Truncated proposals for scalable and hassle-free simulation-based inference

Deistler et al., 2022, NeurIPS 2022 · [arXiv:2210.04815](https://arxiv.org/abs/2210.04815)

**Evidence: B.** Peer-reviewed (Neural Information Processing Systems); 78 citations in 4.0 yr
(Semantic Scholar, 2026-10-19).

## What they do

- Sequential NPE for one x_o. In round r, draw θ from the prior, but reject it unless q_φ(θ|x_o) > τ, where τ is the ε-quantile of q-densities of posterior samples (HPR_ε).
- The proposal is then proportional to the prior on the true-posterior support. Plain maximum-likelihood NPE loss on data pooled from all rounds therefore converges to p(θ|x_o) exactly at x_o (proof §6.2). There are no atomic or APT loss corrections and no leakage.
- **SBCC:** expected-coverage (HPD) diagnostic. Take the rank of log q(θ*|x*) among log q(θ_j|x*), which is SBC with T = log q. It is cheap because q is a flow with exact density, and runs after every round.

## What they find

- **Benchmarks:** six sbibm tasks, 1k/10k/100k simulations. TSNPE ≈ APT and both beat NPE. APT is slightly better on Gaussian-linear and SLCP; TSNPE is better on SIR and Lotka–Volterra.
- **Truncation error:** usually fewer than 0.1% of true-posterior samples fall outside HPR_ε. Results are insensitive to ε ≤ 1e-4; the values tried were 1e-3, 1e-4 and 1e-5.
- **Pyloric network** (31 parameters, 18 summary statistics, 13 rounds × 30k simulations):
  - APT samples inside the prior bounds fell to 0.02% after round 2 and to 0.0000% after round 3.
  - A sigmoid-constrained APT instead piles mass in spikes at the bounds.
  - TSNPE completed all 13 rounds without trouble.
  - Joint truncation rejects 99.94% of prior draws, against 20% for marginal truncation (TMNRE-style).
- **L5PC neuron model** (20 parameters, 35 summary statistics, 6 rounds × 30k simulations):
  - 99.98% of prior predictives have an undefined summary statistic.
  - TSNPE works. APT had 0 of 10M samples inside the prior bounds after round 2.
- **SIR proposal:** with K = 1024, ESS averaged about 25 and was never below about 2.5.

## Implementation details

- **Threshold:** τ is the ε-quantile of log q over posterior samples. Use ε = 1e-3 to 1e-4. A large ε (0.1) cuts the tails (Fig. 11). This is not recommended when tails matter.
- **Proposal sampling:** use rejection sampling from the prior by default. Switch to SIR from q when the acceptance rate collapses (K = 1024; small K leads to narrowing and divergence over rounds, Fig. 13). Monitor the ESS.
- **Pooling and SBCC:** pool data from all rounds; it is still exact at x_o. When pooling, draw SBCC θ* from the pooled mixture proposal, not from the last round's (§6.7).
- **Coverage at high levels:** what matters is the coverage at high confidence levels, since overconfidence there means the truth gets truncated.
  - If round 1 is overconfident, train an ensemble of 10 density estimators to broaden it (L5PC). The final result is slightly underconfident, which is acceptable.
- **Invalid simulations:** do not drop NaN or invalid simulations. Replace them with a value far from the data (the minimum of prior predictives minus 2σ, or picked by hand for heavy tails) and train on everything (§6.5, §6.13–6.14).
- **Leakage checks** when using APT (§6.5): look for a lot of mass at low prior density, and for samples piled at transformed bounds.
- **Settings:** batch size 500 instead of 50; NSF density estimator with sbi defaults.

## Caveats

- Per-observation, not amortised: the network is retrained for each x_o, and coverage only holds near x_o.
- Needs a tractable density q for the HPR threshold. For FM that means an ODE log-density, about 200× the cost of a normalising flow (cf. 2410.21477 / Dax+23).
- If q misses a mode, the truncation removes it for good, so SBCC must be checked.
- Errors can accumulate over many (≫10) rounds when the true posterior has unbounded support.
