# Flow Matching on General Geometries

Chen et al., 2023, ICLR 2024 · [arXiv:2302.03660](https://arxiv.org/abs/2302.03660)

**Evidence: A.** Peer-reviewed (International Conference on Learning Representations); 310
citations in 3.7 yr (Semantic Scholar, 2026-10-19).

## What they do

Riemannian CFM regresses v_t(x_t) ∈ T_x M onto the conditional velocity dx_t/dt under the metric norm ||·||_g.
- The conditional flow is defined through a premetric d. It shrinks d(x_t, x1) = κ(t) d(x0, x1), with u_t(x|x1) = (d log κ/dt) d ∇d/||∇d||_g², which is the minimum-norm such field (Thm 3.1).
- With d the geodesic distance and κ = 1-t, x_t = exp_{x1}(κ(t) log_{x1}(x0)) is the constant-speed geodesic, so training is simulation-free on "simple" manifolds (sphere, flat torus, hyperbolic, SPD).
- General manifolds and meshes use spectral premetrics (biharmonic, diffusion distance). These require ODE simulation of x_t.

## What they find

- Test NLL on earth datasets on S² (Table 2): Volcano -7.93 vs -6.61 for the best prior method (Riemannian diffusion); Fire -1.86 vs -1.40 (CNF Matching). Earthquake (-0.28 vs -0.40) and Flood (0.42 vs 0.25) are not better than the best baseline.
- Protein and RNA torsion angles on the torus (Table 3): RNA 7D NLL -5.20 vs -3.70 for Riemannian diffusion. The 2D protein sets are roughly tied (within error).
- High-dimensional tori (wrapped Gaussian, scale 0.2): RFM log-likelihood stays flat as dimension grows to the hundreds. Riemannian score-based (ISM) and Moser flow degrade.
- Meshes: biharmonic works out of the box, while the diffusion distance needs τ tuned.
- Runtime: simulation-free training runs about 17x faster than 200-step simulation (104 vs 6.4 it/s on a V100).

## Implementation details

- **Parameterization:** the network outputs an ambient vector, which is projected to the tangent space at the projected point: v = g^{-1/2} P_{π(x)} v_θ(t, π(x)). The g^{-1/2} normalization "bypasses manifold-specific initialization and leads to more stable training".
- **Eval accuracy:** sphere NLL uses dopri5 at tol 1e-7. They check that samples stay on the manifold. Mesh paths stop at t = 1 - 1e-5 to avoid the 1/(1-t) singularity.
- **Torus representation:** angles on a flat torus, isometric to the product of circles.
- **Training setup:** MLPs (e.g. 3×512); 50k iterations at batch 512 for the tori; early stopping on validation NLL.

## Caveats

- Unconditional density estimation only, with no conditional or posterior tasks.
- No ablation of the κ scheduler (κ = 1-t throughout), of the time sampling, or of ambient vs intrinsic parameterization.
- The geodesic path is ill-defined at the cut locus (antipodes and half-turns on circles); the paper does not discuss it.
- The gains are mainly on concentrated or high-dimensional data.
