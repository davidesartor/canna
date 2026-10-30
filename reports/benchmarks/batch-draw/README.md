# Batch draw: whether simulating the training batch slows LISA training (2026-10-24)

> **Timed on the 10-22 network** (measured 2026-10-24, src at 54f11d7 or earlier, before the catalogue
> prior, unordered sources, the x1 head and the per-source frame). The scripts follow HEAD and were
> not rerun. At HEAD `paths()` draws the base draws with `jr.normal` and x1 by `physical_to_flow`, so
> a batch is about 1 jaxgb pass per observation, plus 64 single-source candidate passes per
> observation (`sample_window`, `visible_snr = 7` by default). The numbers below are the
> 10-22 ones.

## Takeaways

- **Drawing the batch matters only for small full-band networks.**
  - Inside the trainer's step on the A100 80 GB the full-band draw takes 41 ms: 11% of full-B's
    step and 31% of full-XS's (41% on the L40S). In the narrow band it is under 5% from size S up.
- **The base draws carried the cost (10-22 network only).** 16 of a training observation's 18 jaxgb
  passes were the base draws' SNR rescaling, so a training batch of 256 cost 6.8× its bare
  observations for full-B on an A100. At HEAD the base draws need no jaxgb pass.
- **Keep the draw on the GPU.** 8 CPU cores take 0.28 s for a narrow-B batch of 256, 2.1× the
  A100 80 GB's whole narrow-B step, and 6.5 s for full-B, 17× its step.

## What the page shows

[batch-draw.html](batch-draw.html). The top bar picks a band, a training batch or bare
observations, 1, 4 or 16 sources, and the trainer card whose step is overlaid (A100 80 GB, L40S).

**Tiles**: the draw inside the trainer card's band-B step at batch 256, its share of that step,
the training batch's cost over bare observations on the A100 40 GB, and the CPU's training batch
against the step.

1. **How long does a draw take, against the batch?** The simulator alone per device (A100 40 GB,
   GTX 1080 Ti, 8 CPU cores) at the band's size B. For a training batch of 4 sources the draw
   inside the trainer card's step is a diamond and the step a horizontal line.
2. **How much of the trainer's step is the draw?** The draw's share per config at batch 256, per
   trainer card.
3. **What this means**: the takeaways, short.

The state lives in the URL hash, e.g. `#band=full&kind=train_sample&src=4&card=a80`.

## Method

**Simulator** ([`bench_simulator.py`](scripts/bench_simulator.py),
[`simbench.sbatch`](scripts/simbench.sbatch)).
- narrow-B and full-B at 1, 4 and 16 sources, each size one jitted vmapped call, best of 5 after a
  warm-up. jaxgb runs in float64.
- A **training batch** is `train_sample` with 16 draws per observation: 18 jaxgb passes per
  observation on the 10-22 network, about 1 at HEAD. A **bare observation** is `sample_window`,
  `sample_physical`, `sample_observation` and `preprocess`: 2 passes on the 10-22 network (the
  script now draws windows with `sample_window`, as HEAD).
- Jobs: 66002206 (A100-PCIE 40 GB), 66002207 (GTX 1080 Ti), 66002223 (8 CPU cores).

**Inside the step.** The draw's time inside the trainer's fused epoch (`draw_s`) comes from
[step-time-memory](../step-time-memory/)'s step check: the 10-22 network, batch 256 observations, bf16 backbone,
on the A100-SXM4 80 GB and the L40S.

## How to rebuild

The page needs only the standard library and runs on a login node:

    uv run --frozen python reports/benchmarks/batch-draw/scripts/build_page.py

It reads `data/` and `../step-time-memory/data/*-b256.jsonl`, inlines everything, and prints every
number the takeaways quote.

To remeasure: the `scripts/simbench.sbatch` header.

## Data

- `data/{a100,1080ti,cpu}.json`: per device, `records` of `band`, `kind`, `sources`, `n`, `ms`,
  `compile_s`. The `get_tdi` records (the bare jaxgb response) are not on the page.

The paper reads `data/` in `paper/scripts/numbers_training.py` and `fig_overhead.py`.
