# Score-Based Generative Modeling through Stochastic Differential Equations

Song et al., 2020, ICLR 2021 · [arXiv:2011.13456](https://arxiv.org/abs/2011.13456)

**Evidence: A.** Peer-reviewed (International Conference on Learning Representations); 12610
citations in 5.9 yr (Semantic Scholar, 2026-10-19).

## What they do

- Forward SDE to noise, with a reverse-time SDE driven by the score ∇log p_t. VE and VP SDEs unify SMLD and DDPM.
- Predictor-corrector samplers: an SDE-solver step followed by Langevin MCMC correction.
- The probability-flow ODE shares the marginals and gives exact likelihoods.
- Controllable generation adds ∇log p_t(y | x_t), from a time-dependent classifier or a heuristic.

## What they find

- CIFAR10: FID 2.20, IS 9.89 (NCSN++ deep, VE); 2.99 bits/dim (VP, ODE).
- PC samplers beat predictor-only samplers at equal compute.
- First 1024² score-based samples.

## Implementation details

Langevin correctors at fixed t fix the marginals from accumulated discretisation error. The ODE gives exact likelihoods via the Hutchinson trace.

## Caveats

many NFEs. Controllable generation needs a known or trained p_t(y | x_t).
