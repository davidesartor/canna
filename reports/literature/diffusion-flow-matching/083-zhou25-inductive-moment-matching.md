# Inductive Moment Matching

Zhou et al., 2025, ICML 2025 · [arXiv:2503.07565](https://arxiv.org/abs/2503.07565)

**Evidence: A.** Peer-reviewed (International Conference on Machine Learning); 125 citations in
1.6 yr (Semantic Scholar, 2026-10-19).

## What they do

IMM learns a one-step sampler from time t to s < t, defined through a deterministic DDIM interpolant (which is self-consistent and therefore marginal-preserving). It is trained by *induction*: the push-forward distribution from t is matched, in MMD, to the stop-gradient model's push-forward from a slightly closer r(s, t). Groups of M particles share each (s, r, t), so the loss is an M×M kernel matrix. MMD with a Laplace kernel matches all moments. Single-particle (M = 1) training with an energy kernel reduces exactly to a consistency model, which the authors use to explain why CMs are unstable. Theorem 1 gives convergence in distribution.

## What they find

- CIFAR-10 (unconditional): 3.20 at 1 step, 1.98 at 2 steps (sCT 2.97 / 2.06).
- IN256 XL/2, w = 1.5: 8.05 at 1 step, 3.99 at 2, 2.51 at 4, 1.99 at 8, 1.90 at 16 (DiT-XL/2 at 250×2 NFE: 2.27).
- Particle count M: training *collapses* at M = 1 (CM) and M = 2 on IN256. M = 4 is best per unit compute; larger M samples fewer distinct t per batch.
- Parameterisation (2-step FID):
  - Identity (direct x-prediction) is worst: 46.4–47.3 on IN256, vs about 27 for skip-style parameterisations.
  - Euler-FM (c_skip = 1, c_out = −t·σ_d) is best at scale (27.01).
- Weighting, IN256 FID 40.19 → 27.43 in steps:
  - The ELBO weight alone *hurts* (96.43).
  - Adding α_t brings it to 33.44.
  - Adding 1/(α² + σ²) brings it to 27.43.
- Mapping gap: r(s, t) is a constant decrement in η = σ/α of (η_max − η_min)/2^k. k from 11 to 13 is stable; k = 14 becomes unstable.

## Implementation details

- Laplace kernel exp(−w̃·max(||x − y||, ε)/D) with w̃ = 1/|c_out(s, t)|; it gives better gradients than RBF.
- Reuse x_t to construct x_r (variance reduction).
- c_noise(t) = 1000·t. The result is insensitive to c, but a large c separates nearby r and t.
- Fourier embeddings with scale 16 are stable here, unlike in CMs.
- Pushforward sampler with uniform t decrements.

## Caveats

- Needs M ≥ 4 particles per (s, t), so the batch must be grouped.
- 1-NFE quality is weak (8.05); later TVM, MeanFlow and α-Flow beat it at 1–2 NFE.
- The theory assumes infinite capacity.
