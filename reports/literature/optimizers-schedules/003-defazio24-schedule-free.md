# The Road Less Scheduled (Defazio et al., 2024)

[arXiv:2405.15682](https://arxiv.org/abs/2405.15682), NeurIPS 2024. The original schedule-free method.

**Evidence: A.** Peer-reviewed (Neural Information Processing Systems); 213 citations in 2.4 yr
(Semantic Scholar, 2026-10-19).

## What they do

- Replace the lr schedule with averaging. Three sequences:
  - z takes the base optimizer's steps at a constant lr;
  - x is a running average of z, each step weighted by lr² (so warmup steps count less);
  - the gradient is taken at y = βx + (1−β)z, with β ≈ 0.9.
- Evaluate at x. It is usable at any step, with no horizon fixed in advance.
- SF-AdamW drops Adam's inner momentum (β does that job) and applies weight decay at y. The
  optax version (`optax.contrib.schedule_free`) does the same: it passes y to the base
  optimizer, so a base with decay decays at y.

## What they find

- Matches or beats tuned cosine/linear-decay schedules across 28 problems, from convex
  regression to ImageNet and LLMs.
- Won the self-tuning track of AlgoPerf 2024. Its strength is robustness without tuning more
  than peak quality.
- Needs an lr warmup. Breaks BatchNorm statistics, which must be recomputed at x.
- Best results need unusually small weight decay.

## Limits

- Experiments are at small batch, on V100s. The large-batch and long-run failures were found
  later (ScheduleFree+, SF-NorMuon).
- No diffusion or flow-matching experiments.
