# Stochastic Interpolants: A Unifying Framework for Flows and Diffusions

Albergo, Boffi, Vanden-Eijnden, 2023, arXiv; later JMLR · [arXiv:2303.08797](https://arxiv.org/abs/2303.08797)

**Evidence: A.** Peer-reviewed (Journal of machine learning research); 1078 citations in 3.6 yr
(Semantic Scholar, 2026-10-19).

## What they do

x_t = I(t, x0, x1) + γ(t)z with γ(0) = γ(1) = 0, γ > 0 inside, z ~ N(0, I), and (x0, x1) from any coupling ν. The density ρ(t) is the γ = 0 density convolved with N(0, γ²I), so it is smooth and has full support even when ρ0, ρ1 do not.
- Velocity b = E[∂_tI + γ̇z | x_t] (Thm 7, loss (2.13)); score s = −γ⁻¹E[z | x_t] (Thm 8); denoiser η_z = E[z | x_t]. b = v − γγ̇s.
- The same ρ(t) is sampled by the ODE (ε = 0) or by forward SDEs with drift b + εs and any diffusion ε(t). ε changes paths, not marginals.
- Likelihood control (Lemmas 21–22, Thm 23): for the ODE, KL(ρ1‖ρ̂1) is not controlled by the velocity error alone (it also needs Fisher divergence); for the SDE, KL ≤ (1/4ε)∫E|b̂_F − b_F|², so regressing drifts does control KL. Optimal ε* balances the b and s errors.
- Remark 42: with γ = √(a t(1−t)) (not C¹ at the ends), b at t = 1 contains a term (a/2)·∇log ρ1, so the velocity carries score information at the endpoints.
- Rectification (Sec. 5.3) does not change the map; one-sided interpolants recover score diffusion with time reparametrised to a finite interval.

## What they find

Mostly qualitative.
- 2D checkerboard (4×512 MLP): SDE sampling with ε > 0 beats the ODE; the gap is smallest with γ = √(t(1−t)), and even the ODE with that γ beats γ = 0 (Figs. 7–8, log-density error plots only).
- 128-D 5-mode Gaussian mixture: learning (b, η_z) with a tuned ε > 0 gives the lowest KL of the four combinations (b,s), (b,η), (v,s), (v,η). Small ε over-concentrates modes and under-fills tails; large ε over-spreads (Figs. 10–12).
- Oxford flowers 128², one-sided and mirror interpolants: samples shown, no FID.

## Implementation details

- Antithetic sampling (x_t± = I ± γz for each draw) is "necessary for stable training" of objectives with γ⁻¹(t); otherwise variance diverges at the endpoints. Learning the denoiser η_z instead of s avoids it.
- Learn b directly (one network) if only the ODE is needed.
- SDE sampling with a denoiser: integrate to t_f slightly below 1, then take a final denoising step with ε = 0. Choose ε(t) vanishing near the endpoints to avoid the 1/γ singularity.
- Toys: Adam lr 2e-3 halved every 1500 iterations; batches of 25 base × 400 target × 100 times; Heun sampler.

## Caveats

- No quantitative benchmark against FM or diffusion baselines; evidence is low-dimensional toys plus qualitative images.
- The KL control is for the SDE only; the deterministic flow has no likelihood guarantee from the regression loss.
- Euclidean throughout; adding γz on a manifold needs a manifold heat kernel or tangent-space noise, which the paper does not cover.
