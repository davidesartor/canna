# Posterior Flow Matching

Ma et al., 2026, arXiv preprint · [arXiv:2610.13994](https://arxiv.org/abs/2610.13994)

**Evidence: C.** Not peer-reviewed; 0 citations in 0.1 yr (Semantic Scholar, 2026-10-19). Its
claims are the authors' alone: a hypothesis, not a result.

## What they do

A shared backbone outputs a velocity mean v_θ *and* a per-coordinate log-variance ℓ_θ. It is trained with a diagonal-Gaussian NLL on the ordinary CFM targets V = x1 − x0: ½Σ_i[(V_i − v_i)²/γ_i + log γ_i]. Prop. 4.1: the unique minimiser is v = E[V|x_t] (unchanged FM target) and γ_i = Var(V_i|x_t). The affine map x1 = x_t + (1−t)V turns these into endpoint-posterior readouts at no extra cost: μ = x + (1−t)v and Σ_x1 = (1−t)²Σ_v. The variance also sets a state-dependent diffusion tensor a = λ(t)Σ_θ for a marginal-preserving SDE, with drift v + ½∇·a + ½a·ŝ. The score ŝ = (t v − x)/(1−t) is derived from the velocity, which **requires a Gaussian base N(0,I)** (Eq. 8). Prop. 4.2: the SDE dissipates KL between perturbed and reference marginals, weighted per coordinate by γ.

## What they find

- ImageNet-256: FID FM 25.42 / PFM-ODE 24.45 / PFM-SDE 22.28; IS 63.2 / 65.44 / 68.06.
- ShapeNet Chair, MMD-CD×10³: 12.04 (FM) → 10.50 (SDE); 1-NNA 61.74 → 58.23.
- GEOM-QM9 COV-R: 96.47 (ET-Flow) → 98.37 (SDE); MAT-P: 0.098 → 0.081 Å.
- PushT control at 50 steps: FM 0.872, ODE 0.878, SDE 0.914.
- Ablation H.7.1, same mean model with an isotropic SDE instead of the Σ-shaped one: ImageNet FID 24.31 vs 22.28; PushT 0.882 vs 0.914. The isotropic SDE *hurts* QM9 and Airplane relative to the ODE, so the learned covariance is what matters.
- PFM-ODE quality ≈ FM: the variance head does not damage the mean.
- Predicted variance tracks the mean-squared residual over t. On molecules it gives a lower NLL than a time-only covariance fitted on validation data.
- Noise scale λ0 in λ = λ0(1−t): gains at intermediate λ0, then decline. The SDE costs 1.16–1.33× the ODE time.

## Implementation details

- Clip the log-variance to task bounds: [−1,1] ImageNet, [−3,3] ShapeNet/PushT, [−4,4] QM9.
- ImageNet predicts the endpoint mean directly (x-prediction, App. F).
- The SDE's ∇·a uses exact or Hutchinson-JVP derivatives of γ.
- SDE integration stops at T < 1 because the score has a (1−t)⁻¹ factor.

## Caveats

- Diagonal Gaussian only, so cross-coordinate structure is not captured.
- No calibration test: only variance-vs-residual correlation plots.
- No SBI/NPE experiment; the conditional case is App. E (PushT).
- The SDE half needs a Gaussian source. The NLL/variance half does not: Prop. 4.1 holds for any coupling.
- The NLL's 1/γ weighting changes the effective loss weighting during training. No study of how this interacts with the mean's optimisation.
