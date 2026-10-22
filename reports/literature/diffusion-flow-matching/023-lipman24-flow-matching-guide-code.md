# Flow Matching Guide and Code

Lipman et al., 2024, arXiv tech report / tutorial, FAIR · [arXiv:2412.06264](https://arxiv.org/abs/2412.06264)

**Evidence: A.** Not peer-reviewed; 353 citations in 1.9 yr (Semantic Scholar, 2026-10-19).

## What they do

- Self-contained reference for FM and its generalisations, with the `flow_matching` PyTorch library.
- Losses: any Bregman divergence D gives unbiased CFM gradients, because ∇_v D is affine in its first argument (Thm 4, Prop. 1). More generally, any conditional expectation E[f(X0,X1)|X_t] (x1-, x0-, or velocity-prediction) is learnable by a "conditional matching" regression (Thm 7).
- Time density: sampling t ~ ω(t) is the same as weighting the loss by ω(t) (Eq. 4.27).
- Affine paths: u_t(x) = α̇E[X1|x] + σ̇E[X0|x]. This gives x1-prediction, x0-prediction and velocity, related by the affine maps in Table 1.
  - Converting x1-prediction to velocity, e.g. (x̂1 − x)/(1 − t) for CondOT, has coefficients that blow up as t → 1. In theory the singularities cancel. In practice the learned x̂1 is imperfect, so the guide patches the endpoint with the closed-form u_1(x) = α̇_1x + σ̇_1E[X0|X1 = x].
- Post-training scheduler change (§4.8.2):
  - Any two affine schedulers are related by a scale-time transform: t_r = ρ⁻¹(ρ̄(r)) and s_r = σ̄_r/σ_t, where ρ = α/σ is the SNR. This gives the velocity under the new scheduler in closed form.
  - In theory, all schedulers give the same t = 1 samples.
- Riemannian FM (Sec. 5):
  - Bregman losses are defined on each tangent plane.
  - Geodesic conditional flow: ψ_t(x0|x1) = exp_{x0}(κ(t) log_{x0}(x1)), with any monotone κ.
  - Premetric velocity (Chen & Lipman): u_t(x|x1) = (d log κ̄/dt) d(x,x1) ∇d/‖∇d‖²_g.
  - Antipodal/critical-point singularities have measure zero and are said not to matter in practice. An x-dependent scheduler κ̄(t,x,x1) can remove them.
- Sec. 10: diffusion training is FM with x0- (score-) prediction on a Gaussian path.
  - Stochastic samplers form a family with the same marginals, indexed by β_t. The best β_t trades estimation error against discretisation error and is found empirically (EDM, 2206.00364).
  - Probability-flow ODE sampling is not a true time reversal and does not need to be.

## What they find

none new. It is a tutorial; the code listings (Codes 5–8) show affine paths, x1-prediction training with conversion to velocity, scheduler change, and geodesic training on the sphere with `manifold.proju`.

## Implementation details

- Project the network output onto the tangent space before the loss (`proju`).
- Bregman losses other than MSE are allowed.
- Train x1-prediction with plain MSE to x1, then wrap the model as a velocity model for the ODE solver.
- Patch the t → 1 singularity of the x1→v conversion with the endpoint formula.
- CFG in velocity form: ũ = (1 − w)u(x|∅) + w·u(x|y).

## Caveats

- Derived by me, not in the guide: with d = d_g, (5.19) gives u_t(x|x1) = κ̇/(1−κ) · log_x(x1). The marginal is therefore κ̇/(1−κ) · E[log_{x_t}(X1)|x_t], which on a curved manifold is not log_{x_t}(E[X1|x_t]).
  - A point-valued x1-head therefore converts to the marginal velocity only approximately. The error is large exactly when the posterior over X1 is wide or multimodal on a sphere or circle.
  - The exact manifold analogue of x1-prediction is a tangent "log-prediction" E[log_{x_t}X1 | x_t], which is velocity up to the scalar κ̇/(1−κ).
- Secs. 6–9 were not read line by line.
