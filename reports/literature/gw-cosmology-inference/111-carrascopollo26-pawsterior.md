# Pawsterior: Variational Flow Matching for Structured Simulation-Based Inference

Carrasco-Pollo et al., 2026, arXiv · [arXiv:2602.13813](https://arxiv.org/abs/2602.13813)

**Evidence: C.** Not peer-reviewed; 2 citations in 0.7 yr (Semantic Scholar, 2026-10-19). Its
claims are the authors' alone: a hypothesis, not a result.

## What they do

Variational flow matching (VFM, Eijkelboom et al. 2024) applied to NPE.
- For an affine path x_t = α_t x0 + β_t x1, the marginal velocity is u_t = a_t x_t + c_t E[x1|x_t], with a_t = α̇/α and c_t = β̇ - a_t β.
- Since E[x1|x_t] lies in the convex support Ω, the velocity lies in an affine image of Ω. The paper calls this "endpoint-induced affine confinement".
- **Pawsterior (two-sided VFM):** predict mean-field endpoint distributions for both x0 and x1 given (θ_t, x). Train by endpoint log-likelihood (MSE for continuous blocks, cross-entropy for categorical ones). Form v = α̇ μ0 + β̇ μ1. This avoids the 1/(1-t) or 1/α_t division of one-sided endpoint prediction.
- Bounded parameters: the network output passes through a tanh into [low, high]. Categorical parameters are trained in logit space and set by softmax, then argmax at the end of sampling.

## What they find

- sbibm, 1e3/1e4/1e5 simulations, C2ST:
  - Pawsterior beats FMPE on most tasks. The largest gains are on bounded-support posteriors, and there are gains on unbounded tasks too.
  - Values appear only in Fig. 2, never in the text, so no headline numbers can be quoted.
- Switching Gaussian mixture task (purely categorical, K=10 regimes, T=10):
  - FMPE stays at C2ST close to 1.0 even with 1e5 simulations.
  - Pawsterior falls to about 0.6, improving with both data and depth.
- One-sided vs two-sided VFM: the comparison exists only on SGM (App. C, figure only). The authors report two-sided as "more stable" but never ran one-sided on sbibm.

## Implementation details

- Setup:
  - Network: ResNet MLP, hidden 512–1024, 15–18 blocks.
  - Optimizer: Adam at 1e-3 or 1e-4, gradient-norm clipping at 1, ReduceLROnPlateau, batch 1024.
  - Precision and sampling: bf16 AMP, 100 Euler steps.
- Time prior is a power law, p(t) ∝ t^α with α ∈ {-0.5, -0.25, 0, 1, 4}, tuned per task. Late-t emphasis helps when bounds are sharp.

## Caveats

- The sbibm improvement comes from a hyperparameter sweep, and the gains are reported only as a figure.
- Two things are confounded: the effect of confinement versus the parameterisation change (endpoint prediction with a tanh head).
- No calibration metric (SBC or TARP).
- The categorical task is synthetic.
- Ω must be convex. Circles and spheres are not convex sets, so for those the endpoint mean leaves the manifold.
