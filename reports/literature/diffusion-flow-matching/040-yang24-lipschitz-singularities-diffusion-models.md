# Lipschitz Singularities in Diffusion Models

Yang, Feng, Zhang, Shen, Zhu, Huang, Zhang, Liu, Zhao, Zhou, Cheng; arXiv v2 Dec 2024, ICLR 2024 · [arXiv:2306.11251](https://arxiv.org/abs/2306.11251)

**Evidence: B.** Peer-reviewed (International Conference on Learning Representations); 30
citations in 3.3 yr (Semantic Scholar, 2026-10-19).

## What they do

- **Thm 3.1:** with ε_θ = −σ_t∇log q_t, ∂ε/∂t = (α_t/√(1−α_t²))(dα_t/dt)∇log q_t − σ_t ∂_t∇log q_t. As t → 0 with dα/dt|₀ ≠ 0 (true for linear, quadratic and cosine), limsup‖∂ε/∂t‖ = ∞. Gaussian example: ∇log q = −x, so ∂ε/∂t ∝ dσ/dt → ∞.
- **For v-prediction:** v = −(σ_t/α_t)(x + ∇log q_t), with d(σ/α)/dt → dσ/dt, giving the same blow-up.
- **Empirical estimate:** K(t,t′) = E‖ε_θ(x_t,t) − ε_θ(x_t,t′)‖/Δt.
- **E-TSDM:** split [0, t̃) into n equal sub-intervals and condition on the left endpoint, f_T(t) = max{t_{i−1} ≤ t}; f_T(t) = t for t ≥ t̃. The forward process, loss and sampler are unchanged.
- **Thm 4.1:** ‖ε*(x, f_T(t)) − ε(x,t)‖ ≤ σ_t̃ K(x)Δt + B(x)Δσ_max, which is O(Δt^{1/2}).
- Defaults: t̃ = 100 of T = 1000, n = 5.

## What they find

- **FFHQ-256, FID-10k, Table 2 (DDPM baseline → E-TSDM):**
  | Schedule | Discrete | Continuous |
  |---|---|---|
  | linear | 9.50 → 6.62 | 9.53 → 6.95 |
  | quadratic | 13.79 → 9.69 | 14.26 → 9.66 |
  | cosine | 27.17 → 26.08 | 25.65 → 16.80 |
  | cosine-shift | 14.51 → 11.20 | 12.80 → 9.94 |
  | zero-terminal-SNR | 11.66 → 8.58 | 10.89 → 8.96 |
- **FFHQ FID-50k:** DDPM 6.88, E-TSDM 5.21, E-TSDM-large 4.22.
- **Fast samplers (FID-10k, DDPM → E-TSDM):** DPM-Solver-3 at 20/50 NFE: 21.91 → 16.97 and 24.48 → 13.52. DDIM at 50/200: 21.80 → 19.34 and 23.16 → 13.71. The baseline gets *worse* with more NFE; E-TSDM does not.
- **v-prediction:** FFHQ 10.85 → 9.00; LSUN-Cat 18.40 → 13.86.
- **Other models:** LDM 4.98 → 4.61 (n = 20); EDM on CIFAR-10 1.904 ± 0.015 → 1.797 ± 0.016.
- **Super-resolution PSNR:** 24.64 → 25.61.
- **Ablations (Fig 5):** n = 2 and n = 100 both worse; FID degrades as t̃ grows.
- **Alternatives tried:**
  - Regularising ∂ε/∂t: slight gain, costly.
  - Modifying the schedule so dα/dt|₀ = 0: unstable.
  - Remapping the conditioning to λ = 1/t or logit, sampled uniformly in t or in λ: not helpful.

## Implementation details

- Piecewise-constant (quantised) time conditioning only in the stiff end interval; leave the rest continuous.
- Measure K(t,t′) on a trained net to locate where the time-embedding derivative is huge.

## Caveats

- FID-10k on 256² unconditional images only; not verified at text-to-image scale.
- Quantising t forces the net to average over a range of noise levels. The authors note "unknown and potentially harmful effects".
- The mechanism needs dσ/dt → ∞ (VP), which linear-interpolant flow matching avoids by construction.
