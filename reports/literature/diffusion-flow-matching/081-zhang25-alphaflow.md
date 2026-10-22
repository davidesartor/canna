# AlphaFlow: Understanding and Improving MeanFlow Models

Zhang et al., 2025, arXiv; ICLR 2026 submission per its LLM-usage appendix · [arXiv:2510.20771](https://arxiv.org/abs/2510.20771)

**Evidence: B.** Not peer-reviewed; 50 citations in 1.0 yr (Semantic Scholar, 2026-10-19).

## What they do

The MeanFlow loss decomposes into **trajectory FM**, ||u(z, r, t) − v||² with random r, plus **trajectory consistency**, a (t − r)-weighted continuous CT term with *no boundary condition*. The two gradients are strongly anti-correlated (cosine < −0.4 for more than 95% of training). MeanFlow's 75% of r = t samples help because they reduce the TFM loss while interfering less with the TC term. The α-Flow loss α⁻¹||u(z_t, r, t) − (α·ṽ + (1 − α)·u_θ⁻(z_s, r, s))||² with s = αr + (1 − α)t unifies the methods: α = 1 gives TFM, α = ½ gives Shortcut, and α → 0 gives the MeanFlow gradient. A sigmoid curriculum then anneals α from 1 to 0.

## What they find

- IN256, 240 epochs:
  | model | 1-NFE FID | 2-NFE FID |
  |---|---|---|
  | α-Flow-XL/2 | 2.95 | 2.34 |
  | MeanFlow-XL/2, their reproduction | 3.47 | 2.46 |
  | α-Flow-B/2 | 5.40 | — |
  | MeanFlow-B/2 | 6.04 | — |
- XL/2+ (+60 epochs at batch 1024): 2.58 / 2.15, and 1.95 with class-balanced sampling.
- Schedule ablation, B/2 1-NFE FID:
  - Constant α = 0 (MeanFlow): 44.4.
  - Sigmoid 150K→250K: 41.3.
  - Longest transition, 0→400K: 40.0.
- FM-ratio ablation:
  - α-Flow is best at 25% r = t (40.0).
  - MeanFlow is best at 75% (43.1).
  - More r = t samples help 2-NFE and hurt 1-NFE.
- Discrete-α design space (Table 5):
  - Only ṽ = v_t with a *non-EMA* θ⁻ converges (59.2). The EMA and u_θ⁻ variants score 188–319.
  - Adaptive weight ω = α/(||Δ||² + c): 49.7, vs 59.2 unweighted and 52.2 with MeanFlow's weight.
  - Optimal fixed α: 5e-3 (46.2). Smaller α gets worse: 1e-3 gives 57.2.

## Implementation details

- Curriculum: α = 1 − sigmoid(γ·(k − mid)/width) with γ = 25, clamped to 0 below η = 5e-3 and to 1 above 1 − η.
- When α > 0 there is **no JVP**, only a second forward pass at (z_s, r, s). The JVP is needed only once α reaches 0.
- Sampling: consistency sampling beats ODE sampling for 2 steps on XL; ODE sampling is better on B/2.

## Caveats

- The B/2 ablation FIDs are 40–60, far from converged.
- Gains over MeanFlow are about 15%.
- There is no theory for why the TFM manifold is "narrow".
