# Neural Posterior Estimation with Differentiable Simulators

Zeghal, Lanusse, Boucaud, Remy, Aubourg, 2022, ICML 2022 workshop paper · [arXiv:2207.05636](https://arxiv.org/abs/2207.05636)

**Evidence: B.** Not peer-reviewed; 20 citations in 4.3 yr (Semantic Scholar, 2026-10-19).

## What they do

Loss L = L_NLL + λ‖∇θ log q(θ|x) − ∇θ log p(θ, x | z)‖². The joint score (prior score + ∇θ log p(x|θ,z) + ∇θ log p(z|θ)) is unbiased for the marginal posterior score, because its expectation over p(z|x,θ) equals ∇θ log p(θ|x), the same argument as Brehmer et al.
- Affine RealNVP couplings fail with score matching: their gradients are not expressive enough.
- They use **smooth normalizing flows** (Köhler et al. 2021) with C^∞ sigmoid-mixture couplings, inverted by Newton iterations with implicit-function gradients.

## What they find

- Two moons (unconditional): converges with about 200 simulations with the score vs about 500 without (10 seeds, NLL metric).
- Lotka-Volterra, 4 parameters, CNN compression to 4-D, C2ST over 20 seeds, compared with SCANDAL (NLE + score):
  - With a **wide** proposal, gradients do not noticeably help.
  - With a **narrow** proposal, they help both NPE and NLE significantly.
  - With fixed initial conditions (a deterministic score), they help more.
- Interpretation: the score informs the posterior **shape**, not its **location**.

## Implementation details

The flow must be smooth (C^∞) for score losses. λ is problem dependent and not ablated.

## Caveats

Workshop-length paper: two toy problems, no λ ablation, no high-dimensional θ.
