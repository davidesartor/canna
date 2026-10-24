# Step time and memory: how long a LISA training step takes, and whether it fits (2026-10-24)

## Takeaways

- **A B-size run is about a day of steps on an A100 80 GB with the bf16 backbone, in either band.**
  - At the default batch of 256 observations (4096 samples), narrow-B takes 650k × 0.136 s = 1.02 d
    and full-B 240k × 0.372 s = 1.03 d. On the L40S they take 1.16 d and 1.61 d.
  - XL takes 5.2 d (narrow) and 4.5 d (full) on the A100 80 GB, and 7.6 d and 8.0 d on the L40S.
  - Restarts and checkpoints add little on top: see [training-overhead](../training-overhead/).
- **The bf16 backbone is not optional.**
  - At batch 256 in the sweep, fp32 is 24× slower than the bf16 backbone for narrow-B (3.08
    against 0.128 s) and 23× for full-B (8.37 against 0.36 s) on the A100 80 GB, and 10× and
    7.5× on the L40S.
  - bf16 attention alone recovers almost none of it (narrow-B 3.08 s, full-B 7.89 s on the A100):
    the dense layers carry the cost.
  - In fp32 the L40S beats the A100, since the fp32 runs ask for true fp32 matmuls (no TF32).
- **Memory decides only at XL.** The trainer gets JAX's default 75% of the card: 59.4 GiB on the
  A100 80 GB, 33.3 GiB on the L40S. At batch 256 everything up to L fits both cards in one pass.
  - A100 80 GB, full-XL: plans 45.2 GiB alone and runs (1.61 s/step). It is borderline in the
    trainer loop (see [training-overhead](../training-overhead/)).
  - L40S, narrow-XL: plans 29.7 GiB alone and runs (1.00 s). Borderline in the trainer loop the
    same way.
  - L40S, full-XL: out of memory in one pass (plan 44.0 GiB). It runs split in two (2.88 s/step).
  - On the full band the batch draw sets the memory. `train_step` draws the whole batch before
    splitting it into micro-batches, so gradient accumulation cannot shrink the draw's plan.

Configs are `<band>-<size>`: band `narrow` (32 response points, 128 window bins) or `full` (1024,
4096), size XS to XL. `total_steps` is 650k for narrow and 240k for full.

## What the page shows

[step-time-memory.html](step-time-memory.html). The top bar picks a card (A100 80 GB, L40S,
GTX 1080 Ti), a precision (bf16 backbone, bf16 attention, fp32), a band, a size and a batch.

**Tiles**, for the selection:
- the step and its source;
- whether it fits in memory, with the plan, the peak and the cap;
- the days of steps to `total_steps`, with no restarts or checkpoints;
- the network's param count.

Each number comes from the best measurement: the step check of today's network first, then the
sweep of the 2026-10-22 network. Where neither exists the tile says "Not measured"; it never
extrapolates.

Each card answers one question, and clicking a point or cell sets the selection:

1. **How long is a step, and how does it grow with the batch?** Every card and precision the
   sweep ran for the config, against the batch. The selection is bold, accumulated points are
   hollow, the step check is a diamond. The y-axis toggles between seconds per step and ms per
   observation.
2. **Does it fit in memory, and what fills it?** XLA's plan against the batch for every
   micro-batch split, with ✗ for out of memory. Also shown: the draw's own plan, the sweep's 95%
   cap, the trainer's 75% cap and the step check's plan.
3. **What has been measured?** The step for every card × precision × config at the selected batch,
   tagged by source, or OOM, or —.
4. **What this means**: the takeaways, short.

The state lives in the URL hash, e.g. `#card=l40s&prec=backbone&cfg=narrow-XL&batch=256&y=step`
(also `theme=light|dark`).

## Method

All on `LisaFlow` with the trainer's own functions.

**Sweep** ([`bench_micro.py`](scripts/bench_micro.py), [`sweep_micro.py`](scripts/sweep_micro.py),
[`micro.sbatch`](scripts/micro.sbatch), submitted by [`submit_step.sh`](scripts/submit_step.sh)).
- Each point times the trainer's fused epoch, `train_epoch`: the draw, then `train_step`'s scan
  over the micro-batches, then the schedule-free update. It runs `max(2, min(10, 1024·micro /
  batch))` steps, best of 3, and records XLA's memory plan and the batch draw's own plan.
- Per band × size the sweep:
  - doubles an unaccumulated batch from 1 until it runs out of memory;
  - then doubles the batch with accumulation;
  - halves the micro-batch on an out-of-memory;
  - stops at a time bound.
- One process runs a whole chip × band, so memory is XLA's plan, an upper bound. The cap is 95%
  of the card, preallocated.
- fp32 uses `slurm/gpu.sh`'s float32 flags (`--xla_gpu_autotune_level=0`,
  `JAX_DEFAULT_MATMUL_PRECISION=highest`). bf16 attention runs only the attention's scores and
  weighted sums in bfloat16. The bf16 backbone (the default) runs everything but the embeddings
  and the head in bfloat16.
- **The sweep ran the frozen network of 2026-10-22**: the last block was full (2.5–7.6% more
  params than today), and the narrow window had 16 response points and 64 bins.

| jobs | chip | precision |
|---|---|---|
| 65893384/85, 65901163/64 | GTX 1080 Ti 11 GB | fp32 |
| 65979394/95 | L40S 48 GB | fp32 |
| 65979392/93 | A100-SXM4 80 GB | fp32 |
| 65979398/99 | L40S | bf16 attention |
| 65979396/97 | A100 80 GB | bf16 attention |
| 65979390/91 | L40S | bf16 backbone (narrow cancelled before narrow-XL) |
| 65979388/89 | A100 80 GB | bf16 backbone |

**Step check** ([`bench_micro.py`](scripts/bench_micro.py), [`sizes.sbatch`](scripts/sizes.sbatch),
2026-10-23).
- It ran today's network at batch 256 with the bf16 backbone, in `slurm/gpu.sh`'s environment
  like the trainer: one process per config, the 75% cap, no preallocation.
- Each config ran 20 steps, best of 3, retried at 2, 4 and 8 micro-batches on out of memory. It
  records the allocator's peak too.
- Jobs: 66002259 (A100 80 GB), 66002260 (L40S), 66002263 (CPU, param counts with
  [`bench_step.py`](scripts/bench_step.py) `--params-only`).

Every sweep and step-check record on an A100 is the SXM4 80 GB card.

**Sizes.** Only `hidden_dim` × `num_heads` × `num_blocks` change. The counts are today's full-band
ones from `data/params.jsonl`; the narrow band's are 6–25k smaller.

| size | D × heads × blocks | params | Muon share |
|---|---|---|---|
| XS | 256 × 4 × 4 | 16.0M | 76.0% |
| S | 384 × 6 × 6 | 47.1M | 87.8% |
| B | 512 × 8 × 8 | 106.1M | 92.8% |
| L | 768 × 12 × 12 | 345.7M | 96.7% |
| XL | 1024 × 16 × 16 | 810.4M | 98.1% |

## How to rebuild

The page needs only the standard library and runs on a login node:

    uv run --frozen python reports/benchmarks/step-time-memory/scripts/build_page.py

It reads `data/` and the trainer's defaults (`src/canna/lisa/config.py`, `configs/band/*.yaml`),
inlines everything, and prints every number the takeaways quote.

To remeasure, use the sbatch headers (each skips records already in its data file):
`scripts/submit_step.sh` for the sweep, `scripts/sizes.sbatch` for the step check.

## Data

- `data/{chip}-{precision}-{band}.jsonl`: the sweep, one line per batch × micro-batch point.
  The keys are `config`, `chip`, `bf16`, `batch`, `micro_batches`, `plan_gib`, `limit_gib`,
  `step_s`, `draw_s`, `draw_plan_gib`, `optimizer_s`, `compile_s`, and `oom` on out of memory.
  The files without a precision are fp32.
- `data/{a100,l40s}-bf16backbone-b256.jsonl`: the step check. It adds `peak_gib` and `hat_steps`.
  [batch-draw](../batch-draw/) reads its `draw_s`.
- `data/params.jsonl`: today's param counts per config.

The paper reads these at build time: `paper/scripts/fig_compute.py`, `numbers_priors.py`,
`numbers_training.py` and `fig_overhead.py`.
