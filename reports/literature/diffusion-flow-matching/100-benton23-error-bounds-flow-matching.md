# Error Bounds for Flow Matching Methods

Benton et al., 2023, TMLR 2024 · [arXiv:2305.16860](https://arxiv.org/abs/2305.16860)

**Evidence: B.** Peer-reviewed (Trans. Mach. Learn. Res.); 98 citations in 3.4 yr (Semantic
Scholar, 2026-10-19).

## What they do

- Thm 1, via the Alekseev–Gröbner formula and Grönwall: W2(π̂1, π1) ≤ ε·exp(∫0¹ L_t dt). Here ε² is the time-integrated L2 velocity error (unit weight) and L_t is the spatial Lipschitz constant of the *learned* field at time t.
- It is time-varying because L_t blows up near the data end when π1 lacks full support.
- Thm 2: for a stochastic interpolant x_t = α_t x0 + β_t x1 + γ_t z with "λ-regular" data, the true field's ∇v = (γ̇/γ)I − Cov_x(Ẋ_t, Z)/γ_t. Hence ∫L* ≤ λ∫|γ̇|/γ + λ^½R∫(|α̇|+|β̇|)/γ, which diverges if γ = 0 at the ends.
- λ-regular means conditioning on noisy observations inflates noise covariance by at most λ. Log-concave data has λ = 1; a Gaussian mixture with modes within R and width σ has λ = 1 + R²/σ².

## What they find

- Thm 4: W2 ≤ C^{λ^½}·ε·(γmax/γmin)^{2λ} + √d·γmin. Optimising γmin gives W2 ≲ d^{2λ/(4λ+2)} ε^{1/(2λ+1)} (Cor. 1).
- PF-ODE case: ∫L* ≤ λ(1 + log(1/γ1)) for VP and λ log(1/γ1) for VE, so W2 ≤ ε(e/γ1)^λ for VP.
- No experiments.
- The authors note the bound is exponential in λ, and λ may scale with d. Stochastic smoothing (Albergo et al.) gives better guarantees, suggesting "some level of Gaussian smoothing is helpful for sampling."

## Implementation details

The theory's prescriptions:
- Smooth the interpolant with γ_t > 0 throughout, e.g. γ_t = 2R√((δ+t)(1+δ−t)).
- Restrict the function class to K_t-Lipschitz fields.
- Optimal end-noise γmin ∝ ε^{1/(2λ+1)}.

## Caveats

- A worst-case bound in Euclidean space. The operator-norm bound is likely loose.
- The assumptions require the learned v_θ's Lipschitz constant; nothing enforces it in practice.
- The error is measured with unit time weighting; a warped clock changes the measure.
