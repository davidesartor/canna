# Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow

Liu, Gong, Liu, 2022, ICLR 2023 · [arXiv:2209.03003](https://arxiv.org/abs/2209.03003)

**Evidence: A.** Peer-reviewed (International Conference on Learning Representations); 4381
citations in 4.1 yr (Semantic Scholar, 2026-10-19).

## What they do

Train v by least squares, min ∫E‖(X1−X0) − v(X_t, t)‖² dt, on the linear interpolation X_t = tX1 + (1−t)X0. The coupling (X0, X1) is arbitrary, usually independent π0×π1.
- The minimiser v(z,t) = E[X1−X0 | X_t=z] "causalises" the interpolation. The resulting flow preserves every time marginal (Thm 3.3) and gives a coupling whose convex transport costs are no larger than the input coupling's (Thm 3.5).
- "Reflow" retrains on the model's own (Z0, Z1) pairs and provably straightens paths: min_k S(Z^k) ≤ E‖X1−X0‖²/K (Thm 3.7).
- Nonlinear extension (Eq. 6): any differentiable interpolant X_t with optional weight w_t. The marginal-preservation property still holds, but transport-cost reduction and straightening are lost. PF-ODE and DDIM are instances of this extension.

## What they find

CIFAR-10 with the DDPM++ U-Net and an RK45 adaptive solver.
- 1-RF: FID 2.58, recall 0.57, 127 NFE. VP-ODE: FID 3.93, 140 NFE. sub-VP ODE: FID 3.16, 146 NFE.
- Reflow hurts full-simulation quality: 2-RF FID 3.36, 3-RF FID 3.96, attributed to error accumulating in v.
- Reflow helps few-step sampling a lot. One-step Euler FID: 1-RF 378, 2-RF 12.21, 3-RF 8.15. With distillation: 6.18, 4.85, 5.21. The distilled VP-ODE reaches only 16.23.
- The paper notes that the "relative OT cost" metric collapses to about 0 in high dimension even for a random network, so it is misleading as a test of optimal transport.

## Implementation details

- Training: EMA 0.999999, Adam lr 2e-4, dropout 0.15.
- Reflow data: 4M generated pairs, then 300k fine-tuning steps per reflow round.
- k-step distillation samples t on the grid {0, 1/k, …}. One-step distillation uses LPIPS.
- Non-parametric toy estimator: kNN with m=100 and RBF bandwidth h=1.
- Stronger L2 regularisation of the network straightens flows on toys.
- **Degenerate X0|X1 makes v^X ill-defined.** If X0 given X1 has no density, v^X can be undefined or discontinuous. The fix is to add independent Gaussian noise ξ to X0, which gives a randomised map. This is directly the case when the base or target is concentrated.
- Style transfer reweights the loss by a feature Jacobian, ‖∇h(X_t)ᵀ(X1−X0−v)‖² (Eq. 20), which changes the minimiser only in a projected sense.

## Caveats

- Evidence is images and toys only.
- Reflow needs simulating the model and accumulates bias. Too many reflow rounds are explicitly not recommended.
- Straightening is only guaranteed for the linear, Euclidean interpolant.
