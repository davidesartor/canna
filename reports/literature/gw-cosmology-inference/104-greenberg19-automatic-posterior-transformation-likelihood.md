# Automatic Posterior Transformation for Likelihood-free Inference

Greenberg, Nonnenmacher, Macke, 2019, ICML 2019 · [arXiv:1905.07488](https://arxiv.org/abs/1905.07488)

**Evidence: A.** Peer-reviewed (International Conference on Machine Learning); 491 citations in
7.4 yr (Semantic Scholar, 2026-10-19).

## What they do

APT, also called SNPE-C, is sequential NPE. Training maximizes the log of the "proposal posterior" q̃(θ|x) ∝ q(θ|x) p̃(θ)/p(θ), so simulations from any proposal p̃ train the true-posterior network q without importance weights, and all rounds can be pooled. For flows, the normalizer uses **atomic proposals**: per x, a set of M atoms (the true θ plus M−1 contrastive θ's from the batch), turning the loss into an M-way softmax classification of which atom produced x. Proposition 1: this recovers the posterior up to the prior support. For MDNs with Gaussian or MoG proposals there is a closed form.

## What they find

Mostly given in figures (MMD, −log p(θ_true)). Qualitative findings: on two moons (10 rounds × 1000 simulations), SNPE-A misses the crescent structure, SNPE-B is slowed by importance weights, SNL's MCMC fails to mix between modes, and APT captures the structure. On SLCP, APT+MAF beats SNPE-A/B and closes more than half the gap to SNL. With added uninformative Student-t output dimensions, SNL drops to roughly the prior while APT degrades only slightly (likelihood estimators spend capacity on irrelevant x dimensions). APT works end-to-end with an RNN on raw Lotka-Volterra series and a CNN on 100×100 RPS images. Atoms: no wall-clock difference between M=10 and M=100 on one GPU.

## Implementation details

MDN with 2×50 tanh layers and 8 components (20 for two moons). MAF with 5 MADEs of 2×50 tanh. M = 100 atoms. **Failure mode: leakage.** A MAF trained with APT puts increasing mass outside a bounded prior support over rounds. The authors handle it by rejection and suggest (but do not test) reinitializing, a separately trained normalized flow, or a logistic output transform.

## Caveats

No systematic ablation over M or MDN vs MAF. Leakage is not quantified. Experiments are low-dimensional (θ ≤ ~10).
