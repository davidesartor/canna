# Normalizing Flows on Tori and Spheres

Rezende et al., 2020, ICML 2020 · [arXiv:2002.02428](https://arxiv.org/abs/2002.02428)

**Evidence: A.** Peer-reviewed (International Conference on Machine Learning); 194 citations in
6.7 yr (Semantic Scholar, 2026-10-19).

## What they do

Exact-likelihood normalizing flows on the circle S^1, the torus T^D and the sphere S^D, plus products of these.
- **Circle diffeomorphisms.** The conditions are f(0)=0, f(2π)=2π, f'>0 and f'(0)=f'(2π), followed by a learnable rotation. Three constructions:
  - Möbius maps h_ω(z) = (1-|ω|^2)/|z-ω|^2 (z-ω) - ω. These form a group, so expressivity comes from convex combinations, not composition.
  - Circular rational-quadratic splines (CS), with the end slopes tied (s_1 = s_K).
  - Non-compact projection (NCP): an affine map on tan(θ/2 - π/2), also a group.
- **Torus.** An autoregressive (coupling) flow of circle transforms. The conditioners take (cos θ, sin θ) inputs so they are periodic.
- **Sphere.** A recursive construction S^D → S^{D-1}×[-1,1] (a cylinder). First an interval spline on the height, then a conditional flow on S^{D-1}. On S^2 this map is volume-preserving, and the density has no singularity provided the interval spline has positive end slopes.
- **Alternative for spheres.** Exponential-map flows f(x) = exp_x(∇φ(x)), with a new radial scalar field φ = Σ (α_i/β_i) exp(β_i(x·μ_i - 1)). Its density costs O(D^3).

## What they find

- **T^2.** Targets are a von Mises, a 3-mode mixture and a correlated von Mises, at varying inverse temperature β. NCP and Möbius are best; CS degrades on very sharp targets (high β).
- **S^2, 4-mode mixture** (Table 1, KL in nats / ESS):
  | Flow | KL | ESS |
  |---|---|---|
  | Recursive Möbius-spline (MS) | 0.05 | 90% |
  | Exp-map, Sei polynomial field | 0.50 | 43% |
  | Exp-map, radial field, N_T=1, K=12 | 0.82 | 42% |
  | Exp-map, radial field, N_T=6, K=5 | 0.19 | 75% |
  | Exp-map, radial field, N_T=24, K=1 | 0.10 | 85% |
  Stacking many simple exp-map layers beats a few wide ones, but the intrinsic exp-map flow never catches the recursive (cylindrical-coordinate) flow.
- **S^3 (SU(2)).** Recursive flow reaches ESS 84%, KL 0.14.

## Implementation details

- Möbius centre: ω = 0.99 ω'/(1+|ω'|).
- Spline slope floor: softplus + 1e-3.
- Conditioners are ReLU MLPs [in, 64, 64, out]; base distribution is uniform.
- Training: Adam 2e-4, 20k iterations, batch 256, trained by reverse KL to an unnormalized target (not from data).
- NCP needs linearized expressions near the endpoints.
- Choice by use: CS when both sampling and density are needed (analytic inverse); Möbius or NCP when only one direction is needed (combinations invert by bisection).

## Caveats

- Only small synthetic targets (D ≤ 6), fitted by reverse KL rather than conditional density estimation.
- Discrete normalizing flows, not flow matching.
- No comparison with the alternative of embedding circles as R^2 unit vectors plus a projection.
