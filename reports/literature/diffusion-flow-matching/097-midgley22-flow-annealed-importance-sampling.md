# Flow Annealed Importance Sampling Bootstrap

Midgley, Stimper, Simm, Schölkopf, Hernández-Lobato, 2022, ICLR 2023 · [arXiv:2208.01893](https://arxiv.org/abs/2208.01893)

**Evidence: A.** Peer-reviewed (International Conference on Learning Representations); 153
citations in 4.2 yr (Semantic Scholar, 2026-10-19).

## What they do

Train a normalizing flow to an unnormalised target p with no samples by minimising the mass-covering $D_{\alpha=2}(p\|q)$ (= variance of IS weights p/q). Estimate its gradient $-E_g[\nabla\log q_\theta]$ with samples from AIS that starts at q and targets $g\propto p^2/q$ — the minimum-variance proposal, concentrated where p is high and q is low, so it finds missing modes. Surrogate loss = self-normalised AIS-weighted NLL of stop-grad AIS samples. Prioritised replay buffer reuses AIS samples, correcting weights by $q_{old}/q_\theta$.

## What they find

40-component 2-D GMM with pathological init: FAB+buffer ESS 61.9%, fwd KL 0.30 (= ML on true samples 0.31); reverse-KL flow, RBD, SNF, CRAFT miss modes (NaN log-lik); $D_{\alpha=2}$ estimated with flow samples alone fails (ESS 0.7%, MAE 99.5%). Alanine dipeptide (22 atoms, internal coords, 12 RQ-spline coupling layers): FAB+buffer **ESS 92.8%** vs ML-on-MD 2.8%, reverse KL 54% (mode-dropping, spurious); Ramachandran KLD 3.4e-3 (2.5e-3 reweighted) vs ML 7.6e-3; 100× fewer target evaluations than MD. Buffer roughly doubles ESS (31→62% GMM, 52→93% ALDP).

## Implementation details

very few AIS rungs suffice (1 intermediate β=0.5 with 1 MH step for GMM; 8 rungs with HMC for ALDP); self-normalised weights "greatly improved training stability"; sampling buffer *without* replacement worked better despite bias; L=4–8 gradient steps per AIS pass; theory (factorised toy) says AIS rungs must grow linearly with dimension to keep gradient SNR constant, while naive IS gradient variance grows exponentially. Data-free normalisation of coordinates (minimum-energy config + order-of-magnitude scales); circular coords for dihedrals; filter chirality. Defensive IS and control variates suggested but not tried.

## Caveats

needs an exact-density flow (coupling flows; CNF densities would be expensive); demonstrated only up to 60-ish dims; buffer sampling introduces bias.
