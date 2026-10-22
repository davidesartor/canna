# One Step Diffusion via Shortcut Models

Frans et al., 2024, ICLR 2025 · [arXiv:2410.12557](https://arxiv.org/abs/2410.12557)

**Evidence: A.** Peer-reviewed (International Conference on Learning Representations); 455
citations in 2.0 yr (Semantic Scholar, 2026-10-19).

## What they do

Condition the FM network on step size d as well as t. Then s_theta(x_t, t, d) is the average velocity over [t, t + d]. Train jointly on FM at d = 0 and a binary self-consistency target s(x_t, t, 2d) = stopgrad[s(x_t, t, d) + s(x'_{t+d}, t + d, d)]/2, where x'_{t+d} is the model's own step, not the data interpolant. Single network, single run, no schedule.

## What they find

- DiT-B, equal compute, FID at 128 / 4 / 1 steps:
  - CelebAHQ: shortcut 6.9 / 13.8 / 20.5; FM 7.3 / 63.3 / 280.5; consistency training 53.7 / 19.0 / 33.2; reflow 16.1 / 18.4 / 23.2; progressive distillation 1-step 14.8, the only one better.
  - ImageNet-256: shortcut 15.5 / 28.3 / 40.3; FM 17.3 / 108 / 325; CT 42.8 / 43.0 / 69.7.
  - DiT-XL: 1-step 10.6, 128-step 3.8.
- Robotics (1 step): Push-T 0.87, Transport 0.80, vs a 1-step diffusion policy at 0.12 / 0.00.
- Cost: about 16% more compute than FM with k = 1/4 of the batch on self-consistency. Many-step quality slightly better than plain FM; the authors suggest a regularisation effect.

## Implementation details

- Steps and time sampling: M = 128 smallest steps, giving 8 values of d (powers of 2). Smallest d queries the model at d = 0. Sample t on the multiples of d only. Uniform t ~ U(0, 1) works as well as anything.
- Batch mix: 3/4 FM targets, 1/4 self-consistency targets (FM targets are higher variance).
- Stabilisers: EMA params to generate targets. Weight decay 0.1 is "crucial", stopping the model latching onto noise targets early. CFG only at d = 0, and the CFG scale must be fixed before training.
- Code: JAX (github kvfrans/shortcut-models), AdamW lr 1e-4 constant.

## Caveats

Absolute 1-step FIDs (40 on IN256 DiT-B) are far from later methods (MeanFlow 3.43). The comparison baselines are their own reimplementations (e.g. CD 136.5 looks broken). Gap between 1-step and many-step remains.
