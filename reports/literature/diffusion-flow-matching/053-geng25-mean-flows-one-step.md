# Mean Flows for One-step Generative Modeling

Geng et al., 2025, arXiv; NeurIPS 2025 · [arXiv:2505.13447](https://arxiv.org/abs/2505.13447)

**Evidence: A.** Peer-reviewed (Neural Information Processing Systems); 663 citations in 1.4 yr
(Semantic Scholar, 2026-10-19).

## What they do

The network models the average velocity u(z, r, t) = (1/(t − r))∫_r^t v dτ instead of the instantaneous one. Differentiating the definition gives the MeanFlow identity u = v − (t − r)·du/dt, where du/dt is the total derivative along the flow. One JVP with tangent (v, 0, 1) gives it. The target v − (t − r)·du/dt is stop-gradded and regressed against, so training runs from scratch with no teacher, no EMA target and no discretisation. CFG folds into the target field, so a guided sample costs one NFE.

## What they find

- IN256 1-NFE FID: B/2 6.17, M/2 5.01, L/2 3.84, XL/2 3.43. 2-NFE: XL/2 2.93, XL/2+ 2.20.
- Baselines: Shortcut-XL 10.60; IMM 7.77 (1×2 NFE); iCT-XL 34.24.
- CIFAR-10 1-NFE: 2.92, with no preconditioner.
- Ablations, B/4 for 80 epochs, 1-NFE FID:
  - Share of samples with r ≠ t: 0% (pure FM) 328.9; 25% 61.06 (best); 50% 63.1; 100% 67.3.
  - JVP tangent: the correct (v, 0, 1) gives 61.06; the wrong ones give 268–329.
  - Time embedding: (t, t − r) 61.06; (t, r) 61.75; t − r alone 63.1.
  - (r, t) sampler: lognorm(−0.4, 1.0) 61.06; uniform 65.9.
  - Adaptive-weight power p: 0 (plain L2) 79.75; 0.5 63.98; 1.0 61.06.
  - CFG ω = 3.0 gives 15.53.

## Implementation details

- Pseudocode:
  ```
  u, dudt = jvp(fn, (z, r, t), (v, 0, 1))
  u_tgt = v - (t - r) * dudt
  loss = metric(u - stopgrad(u_tgt))
  ```
- Adaptive weight: w = sg(1/(||Δ||² + c)^p) with c = 1e-3.
- IN256 training: Adam with (0.9, 0.95), constant lr 1e-4, batch 256, **no weight decay**, EMA 0.9999.
- CIFAR training: 75% of samples with r ≠ t, p = 0.75, lognorm(−2, 2).
- JVP cost: written in JAX; 0.045 s/iter for FM vs 0.052 s/iter for MeanFlow, about 16% overhead. The JVP output sits under stopgrad, so it adds no θ-backprop.

## Caveats

- Images only; no likelihood or calibration evaluation.
- The r ≠ t share and p need retuning per dataset.
- The JVP must pass through the whole network. With bf16, attention kernels and `jax.checkpoint` this is doable but needs `jax.jvp` support in every custom op.
- 1-NFE quality still trails multi-step FM.
