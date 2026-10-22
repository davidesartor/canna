# Fine-Tuning Flow Matching via Maximum Likelihood Estimation of Reconstructions

Li, Ding, Li, Li, 2025, arXiv v2 Jan 2026, ICML-format submission · [arXiv:2510.02081](https://arxiv.org/abs/2510.02081)

**Evidence: C.** Not peer-reviewed; 0 citations in 1.0 yr (Semantic Scholar, 2026-10-19). Its
claims are the authors' alone: a hypothesis, not a result.

## What they do

After FM pretraining, unroll the actual sampler (e.g. 10 Euler steps) from $x_0$ and backprop the endpoint error against the $x_1$ it was coupled with (minibatch-OT or "sorted pairs" coupling). Thm 3.1 bounds inference error by $e^{L_u}(\sum\delta\tau+\tfrac12 M\tau^2)$ — training loss amplified by the field's Lipschitz constant, worst at basin boundaries where straight-path FM fields become near-discontinuous. Optional "generalised artificial viscosity" residual stage $\dot\phi=\tilde v(o)+A_0x+\sum A_jf_j(D_jx)$ with Hurwitz $A_j$ (Cayley/Schur parameterisation) makes the flow provably contracting.

## What they find

Robot policies (success rate): Franka Kitchen FM 0.742 → FT-FM 0.811; Push-T 0.684 → 0.751; Robomimic 0.729 → 0.791. Fine-tuning an *early* (1/3-trained) model: Kitchen 0.378 → 0.821, Robomimic 0.314 → 0.755 ("above 30% to over 80% in a few epochs"). GAV improves robustness under input noise (Kitchen Noise(G) 0.655 → 0.704); replacing GAV by an MLP (ablation) gives no robustness gain (0.641). ERA5 FlowNP target log-lik 17.98 → 18.31. CIFAR-10: FID 3.57 → 3.55 after 1000 FT steps (negligible). Toy GMM: errors concentrate on basin boundaries and shrink after FT.

## Implementation details

start FT from a not-fully-converged checkpoint; same or reduced LR (robot: 1e-4 pretrain, 5e-6 FT); 10 Euler steps unrolled, direct backprop (adjoint possible); minibatch-OT pairing in pretraining. Curves overfit and decline after a peak — needs early stopping. Works best for low-dimensional outputs (~a dozen dims) with rich conditioning; struggles at image scale.

## Caveats

The endpoint-MSE objective toward a *paired* sample regresses toward a conditional mean of x1 given x0 — for a genuinely stochastic posterior it shrinks diversity and is not a proper density objective; evaluation uses success rate/FID, not calibration. Gains for generative fidelity (CIFAR) are nil. Theory assumes Lipschitz true field.
