# Riemannian MeanFlow

Woo et al., 2026, ICML 2026 per keywords · [arXiv:2602.07744](https://arxiv.org/abs/2602.07744)

**Evidence: C.** Not peer-reviewed; 9 citations in 0.7 yr (Semantic Scholar, 2026-10-19). Its
claims are the authors' alone: a hypothesis, not a result.

## What they do

Average velocity u_{s,t}(x_s) = log_{x_s}(x_t)/(t−s), with flow map exp_x((t−s)u). There are three regression targets: Eulerian (JVP in s), Lagrangian (JVP in t, plus a cycle-consistency loss) and semigroup (no derivatives). Parametrisations: v-pred, x_t-pred, and x1-pred, where u = log_{x_s}(x̂1)/(1−s) with weight (1−s)/max(1−s, ε), ε∈{0.05, 0.1}.

## What they find

- Promoter DNA at 1 NFE: MSE 0.027–0.031 vs Fisher FM 0.030 at 100 NFE. FFM needs ≥32 steps to match RMF at 1 NFE.
- Protein designability:
  - RMF 0.87 at 10 NFE vs FrameFlow 0.61; at 5 NFE 0.82 vs 0.04.
  - RMF 0.35 at 1 NFE.
  - 100-NFE parity: 0.94 vs 0.93.
- Toy helix on S^{D−1}: **x1-pred stable at D=2048, v-pred degrades with D, x_t-pred fails.**
- Look-ahead reward guidance helps.

## Implementation details

- Adaptive loss weighting with p=0.5 (Eulerian is unstable without it).
- **Low-frequency time embeddings (ω=0.02)** for the time-derivative terms.
- GFM baselines (G-ESD/LSD) did not converge on DNA because of high variance in network derivatives.

## Caveats

The ablation numbers are in an unread appendix. The best objective × parametrisation pair varies by metric.
