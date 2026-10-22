# Amortized Bayesian Mixture Models

Kucharský & Bürkner, 2025, arXiv v3 Mar 2026 · [arXiv:2501.10229](https://arxiv.org/abs/2501.10229)

**Evidence: B.** Peer-reviewed (Statistics and computing); 4 citations in 1.8 yr (Semantic
Scholar, 2026-10-19).

## What they do

Factorise p(θ, z | y) = p(θ | y) p(z | y, θ).
- A normalising flow with summary networks (DeepSet, or LSTM for HMMs) does NPE on the mixture parameters, with z marginalised out of the simulator's data.
- A softmax classifier q(z_i | h(y_i), θ) is trained with cross-entropy on the true indicators. For dependent mixtures (HMMs) it does filtering or smoothing, using forward and backward networks trained with separate losses and combined only at prediction time.
- All networks are trained jointly. At inference, each θ sample is passed to the classifier, which propagates parameter uncertainty into the membership probabilities.

## What they find

- GMM (K=3, N=150–250): C2ST vs Stan = 0.53. SBC looks fine. BayesFlow takes about 12 s for 500 datasets × 1000 draws on CPU; Stan takes about 3 s per dataset.
- HMM (2 states, N=100): C2ST = 0.56. Recovery of the transition probabilities degrades when the states are poorly separated (μ₂−μ₁ < 2/3).
- Real data (Dutilh 2011 EAM switching model): the MMD misspecification check flags 4 of the 5 participants.
  - Participant B: biased but fixable with PSIS.
  - Participants A and G: posterior in the prior tail, Pareto k̂ high, cannot be corrected.
  - Classification is still fine for A even though θ is biased.

## Implementation details

- Ordered-vector reparameterisation via log-differences; stick-breaking logit for simplexes.
- MMD-regularised summary space as an out-of-distribution or misspecification test.
- PSIS correction with a Pareto k̂ gate. This requires a tractable density on both sides.
- Train the forward and backward classifiers separately so neither collapses onto the other.
- Priors chosen for MCMC (flat or order-restricted) can be poor choices for ABI.

## Caveats

- Toy-scale (K ≤ 3).
- Label switching is solved only by construction (ordered, well-separated priors). Nothing is shown for overlapping components, where ordering creates truncation artefacts.
- Normalising flow, not flow matching.
