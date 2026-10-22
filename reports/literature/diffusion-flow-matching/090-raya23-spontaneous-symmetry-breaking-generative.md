# Spontaneous Symmetry Breaking in Generative Diffusion Models

Raya et al., 2023, NeurIPS 2023 · [arXiv:2305.19693](https://arxiv.org/abs/2305.19693)

**Evidence: B.** Peer-reviewed (Neural Information Processing Systems); 97 citations in 3.4 yr
(Semantic Scholar, 2026-10-19).

## What they do

Write the reverse VP-SDE as gradient descent on a potential u(x,t). For a symmetric dataset, the central fixed point (the data mean) is stable early and loses stability at a critical θ_c, where the potential becomes a Mexican hat. Before θ_c, fluctuations are mean-reverted and irrelevant. Around θ_c they are amplified and decide which sample is generated. After θ_c they are pulled back to the chosen branch.

## What they find

- 1-D two-point data ±1: θ_c = sqrt(√2-1) ≈ 0.644. The fixed points obey a Curie–Weiss-type equation.
- Normalised, centred datasets of norm r: the Laplacian at the origin flips sign at θ* = sqrt((√(D²+r⁴)-r²)/D).
  - App. A.3 (hypersphere): the Hessian is isotropic, so this is exactly the critical point.
  - Eq. 41 prints r instead of r², an apparent typo.
- Empirically (DDPM T=1000), FID is flat versus a late start up to a threshold, then degrades sharply.
- The "gls" fix: start late from a Gaussian fit to the noised data. Selected FID results (vanilla → gls):
  | Dataset | Sampler, steps | Vanilla | gls |
  |---|---|---|---|
  | CIFAR10 | DDPM, 5 | 84.82 | 42.46 |
  | CIFAR10 | DDIM, 3 | 109.37 | 42.31 |
  | CelebA64 | DDPM, 10 | 36.66 | 23.79 |
  | CelebA32 | PNDM, 3 | 235.87 | 28.60 |
  | Imagenet64 | DDIM, 10 | 38.21 | 36.25 |
- Gains shrink at 10 steps for the stronger samplers. gls also increases attribute diversity (race entropy on CelebA64).
- The pairwise bifurcation seen in the trained-model potential happens later than the FID-curve threshold, so there is more than one breaking event in real data.

## Implementation details

Skip the mean-reverting early phase by starting at the noised-data Gaussian. Concentrate sampler steps around the instability. The critical time is found empirically by scanning the start time, since it is only analytic in toy models.

## Caveats

This is about sampling, not training. The critical time is only computed for symmetric toy data with a Gaussian base and a VP forward process. Nothing is conditional: the symmetry is broken by noise, not by an observation.
