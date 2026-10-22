# Terminal Velocity Matching

Zhou et al., 2025, arXiv / Luma AI · [arXiv:2511.19797](https://arxiv.org/abs/2511.19797)

**Evidence: B.** Not peer-reviewed; 21 citations in 0.9 yr (Semantic Scholar, 2026-10-19).

## What they do

The paper learns the displacement f(x_t, t, s) = (s − t)·F(x_t, t, s) by matching its velocity at the *terminal* time s rather than the initial time t: d/ds f(x_t, t, s) ≈ u_θ*(x_t + f_θsg(x_t, t, s), s), plus an FM term ||u_θ(x_s, s) − v_s||² on the diagonal F(·, s, s). This is the Lagrangian/LMD condition of 2406.07507, trained from scratch with the model's own diagonal as teacher. With a Lipschitz u_θ, the weighted integral of the loss upper-bounds W₂²(model, data). It is the s-derivative dual of MeanFlow, which differentiates in t. The JVP is only with respect to s, so the high-variance v_t never passes through the JVP, which is why random CFG during training stays stable. Gradients flow *through* the JVP term, which needed a custom flash-attention JVP kernel.

## What they find

- IN256: 3.29 at 1-NFE (w = 2), vs MeanFlow 3.43. With w = 1.75: 2.61 at 2 NFE and 1.99 at 4 NFE, vs DiT-XL/2 2.27 at 250×2 NFE.
- IN512: 4.32 at 1-NFE (sCT-XL 4.33, MeanFlow 5.24) and 2.94 at 4 NFE.
- Ablations (XL/2, 200K steps, 1-NFE FID):
  - EMA rate γ of the target: 0 → 10.24; 0.9 → 5.08; 0.99 → 4.90; 0.999 → 6.04. γ = 0.99 is the default.
  - CFG: constant w = 2 gives 4.81; random gives 5.14.
  - Scaled parameterisation f = (s − t)·w·F: neutral at w = 2, a large help at w = 1.5 (6.04 vs 9.32).
  - Share of t = s samples: 0% is best for 1-NFE (3.72), 20% is best for 2-NFE (2.97). This *contradicts* MeanFlow's 75%.
  - Time sampling: "gap" sampling (logit-normal on t − s with μ = −0.8, σ = 1.0; s ~ LN(−0.4, 1.0)) works best, and sampling s for the FM term separately helps (3.36 vs 3.44). t should be biased towards 1 and s towards 0, but not too far: μ_t = 2, σ = 2 gives 7.88 under truncation.

## Implementation details

- Semi-Lipschitz DiT:
  - RMSNorm QK-norm.
  - Parameter-free RMSNorm in place of LayerNorm.
  - **RMSNorm applied to the adaLN modulation vectors a(t) and b(t)**.
  - Lipschitz init on every linear layer except the time embedding.
  - Without these, activations in the time-embedding layer blow up.
- AdamW with β₂ = 0.95, needed because of the higher-order gradients through the JVP. lr 1e-4, **weight decay 0**, batch 2048.
- A target EMA of 0.99 that is separate from the evaluation EMA of 0.9999.
- The CFG loss is scaled by 1/w².
- Detaching the JVP cuts step time from 0.95 s to 0.69 s at the cost of a biased gradient.
- Embed (t, t − s).

## Caveats

- The W₂ bound needs Lipschitz constants that are not enforced in practice, only "partial control".
- Costs more memory (71 GB vs 47 GB for MeanFlow with the same kernel).
- Images only.
