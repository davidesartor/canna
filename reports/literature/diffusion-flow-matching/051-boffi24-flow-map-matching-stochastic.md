# Flow map matching with stochastic interpolants: A mathematical framework for consistency models

Boffi et al., 2024, arXiv preprint; venue not stated in the HTML · [arXiv:2406.07507](https://arxiv.org/abs/2406.07507)

**Evidence: A.** Peer-reviewed (Trans. Mach. Learn. Res.); 115 citations in 2.4 yr (Semantic
Scholar, 2026-10-19).

## What they do

Learn the two-time flow map X_{s,t}, parameterised as X = x + (t−s)·v_{s,t}(x) so that X_{s,s} = id holds by construction. Four losses:
- LMD (Lagrangian distillation): ∂_t X_{s,t}(I_s) = b_t(X_{s,t}(I_s)). This is new, uses only a time-JVP and needs no spatial gradient.
- EMD (Eulerian distillation): ∂_s X + (∇X)b = 0. Its continuous-time limit is consistency distillation (CD) and the consistency trajectory model (CTM).
- Direct FMM: requires invertibility X_{s,t}(X_{t,s}(I_t)) = I_t and matches ∂_t of that composition to the interpolant velocity İ_t.
- Progressive FMM (PFMM): distils a K-step map into a one-step map.
- Both the Lagrangian and Eulerian losses upper-bound W₂² between teacher and student. The Lagrangian bound carries an exp(∫Lipschitz) factor; the Eulerian bound does not, but it involves ∇X, which can be near-singular.
- Section 3.6: the naive direct Eulerian loss with İ in place of b has the wrong minimiser because of a term quadratic in İ. Consistency models fix this with stopgrad, which gives the right fixed point but no descent guarantee.

## What they find

- CIFAR-10, teacher stochastic interpolant (SI) at FID 5.53 with an adaptive solver:
  | method | FID (T-FID), 2 steps | FID (T-FID), 4 steps |
  |---|---|---|
  | LMD | 7.13 (1.27) | 6.04 (1.05) |
  | EMD | 48.32 (34.19) | 44.35 (30.74) |
  | SI, 2/4-step Euler | 112 | 34.8 |
  | PFMM (no velocity teacher; 8-step FMM teacher at FID 8.44) | 18.35 | 11.14 |
- ImageNet-32, direct FMM with |t−s| < 0.25:
  | N | FMM FID | BatchOT FID |
  |---|---|---|
  | 4 | 16.9 | 38.9 |
  | 8 | 12.6 | 15.6 |
  | 20 | 9.68 | 7.71 |
- The LMD loss converges about 10× faster than EMD on both datasets.
- Checkerboard: the one-step map learned on all of [0,1]² is worse than a 4-step map learned on the strip |t−s| < 0.25 and then progressively distilled.

## Implementation details

- Get ∂_t X in the same forward pass with forward-mode JVP.
- Embed s and t like the usual t embedding and concatenate them.
- Directly learning a one-step map is hard: train on a strip, w_{s,t} = 1(|t−s| ≤ 1/K), then distil progressively.
- The invertibility term is free, because it comes out of the same JVP.
- The 2D model is an MLP 512×6 trained with Adam for 5e4 iterations.

## Caveats

- Weak teachers and small compute; the authors say teachers were "trained with limited compute".
- The FID numbers are far from the state of the art.
- Direct FMM needs the backward map too, which doubles what the network must learn.
