# Training overhead: what restarts and checkpoints cost a LISA run (2026-10-24)

> **Timed on the 10-22 network** (measured 2026-10-24, src at 54f11d7 or earlier, before the catalogue
> prior, unordered sources, the x1 head and the per-source frame). The scripts follow HEAD (the
> trainer's history keys: `catalog_history` of 6, `frame_history`, `fisher_history`, `flow_history`)
> and were not rerun. The batch was the old default of 256 observations (4096 samples); HEAD's
> `batch_size` is 1024 samples.

## Takeaways

- **Restarts and checkpoints cost 1.2–2.3% of each 24 h job at the default `log_interval` of
  1000, with the bf16 backbone.**
  - The A100 80 GB's full-band runs lose 1.05–1.5%, but their restarts are lower bounds (see
    Method).
  - With them a B run takes 1.03–1.04 d on the A100 80 GB and 1.2–1.6 d on the L40S. The steps
    alone are in [step-time-memory](../step-time-memory/).
- **A restart takes 400–705 s before the first step.** Compiles on the job's 2 CPUs are 73–96% of
  it (371–394 s on the L40S, 485–558 s on the A100), the imports 18–127 s (about 125 s on a node
  whose file cache is cold), and the cold restore 0.7–14 s up to L and 11–60 s for XL.
- **An epoch costs 0.26–4.7 s beyond its steps** (diagnostics, history, the save). The async save
  blocks for only 0.09–0.72 s; its background write (1.2–17 s) overlaps the next epoch.
- **The log interval is not worth tuning with the bf16 backbone.** The best interval is 500–5000
  and saves at most 0.6 points over 1000.
- **A slow step changes that.** bf16 attention makes the step 1.5–28 s, so a 1000-step epoch takes
  hours and the time limit kills half of one on average. 1000 then loses 1.5–17% against
  0.9–1.9% at the best interval of 50–200.
- **XL is borderline in the trainer's own process.**
  - A100 80 GB, full-XL: alone it plans 45.2 GiB and runs (1.61 s/step), but it ran out of memory
    (plan 46.3 GiB) in the trainer loop's shared process, after nine other configs.
  - L40S, narrow-XL: alone it plans 29.7 GiB and runs (1.00 s); in its own trainer-loop process it
    ran out of memory in the epoch (plan 29.8 GiB).
  - L40S, full-XL: it runs split in two micro-batches (2.88 s/step), but the trainer loop runs out
    of memory there too.
  - A100 40 GB: full-XS to full-B run in the trainer loop; full-L and full-XL run out of memory in
    the epoch, each in its own process.

Configs are `<band>-<size>`: band `narrow` or `full`, size XS to XL. `total_steps` is 650k for
narrow and 240k for full.

## What the page shows

[training-overhead.html](training-overhead.html). The top bar picks a card (A100 80 GB,
A100 40 GB, L40S), a precision (bf16 backbone, bf16 attention), a band and a size. Sliders set the
log interval and the total steps (it resets to the band's `total_steps` when the band changes).

**Tiles**, for the selection: the loop's step, the restart and its compile share, the share of a
24 h job lost at the interval, the best interval, the wall time with restarts and saves, and the
checkpoint size. An out-of-memory record says where it ran out.

Each card answers one question over every trainer-loop series of the config; clicking a curve, bar
or cell sets the selection:

1. **How much of a 24 h job is lost, and at which log interval?** The lost share against the
   interval, the best interval as a dot.
2. **Where does a 24 h job go besides steps?** The margin, restart, per-epoch costs and killed
   epoch at the chosen interval.
3. **What does a restart cost?** uv, imports, restore and compiles, with ≥ on lower bounds.
4. **What has been measured?** The lost share and restart for every series × config, or OOM, or —.
5. **What this means**: the takeaways, short.

The state lives in the URL hash, e.g. `#card=l40s&prec=backbone&cfg=full-B&L=1000` (also `T`).

## Method

[`bench_overhead.py`](scripts/bench_overhead.py), [`overhead.sbatch`](scripts/overhead.sbatch),
[`submit_overhead.sh`](scripts/submit_overhead.sh), model in [`overhead.py`](scripts/overhead.py).

- The resources are `slurm/lisa.sbatch`'s: 2 CPUs and `srun uv run`, batch 256 observations (the old default).
- **Restart:** the `uv run` start; the imports, the git check and `setup`; a cold restore from
  `outputs/` with its pages evicted; and the diagnostics and epoch compiles, plus their first
  calls beyond a warm call.
- **Loop:**
  - First, three 5-step epochs, whose saves wait out the previous write. They give the write's
    length in the loop.
  - Then three long epochs. They give how long the save blocks and how much the write slows
    training.
  - The step is the best long epoch per step.
- **The chunk model.** One chunk of T = 24 h loses:
  - the 300 s requeue margin;
  - the restart R;
  - n per-epoch costs c, with n = (T − 300 − R)/(L·s + c) − ½;
  - half the killed epoch, or the whole of it when the kill lands inside its background write.

  c is the sum of the diagnostics, `history.npz`, the save's block, the slowdown during the write,
  and any wait for a write still in flight. The best interval is searched over 1, 2 and
  5 × 10^k.
- The first jobs ran every config in one process. That lent device memory and compiles across
  configs. The L40S full band and the out-of-memory configs were rerun one per process
  (`*-alone.jsonl`); a later job supersedes an earlier record of the same config and card.
- **The A100 80 GB full band was never rerun alone.** Its records followed the narrow configs in
  one process, so their restarts (363–372 s, with 232–240 s of compiles) are lower bounds. The
  bf16-attention full-B and narrow-XL records are lower bounds for the same reason.

| job | chip | precision | configs |
|---|---|---|---|
| 66002249 | A100-SXM4 80 GB | bf16 backbone | all ten, one process |
| 66011762 | A100-PCIE 40 GB | bf16 backbone | full XS, S, B, L, XL, each alone |
| 66003330 | A100-PCIE 40 GB | bf16 attention | narrow-B, full-B, narrow-XL, full-XL, one process |
| 66002272 | L40S | bf16 backbone | all ten, one process |
| 66011274 | L40S | bf16 backbone | full-B, full-L, narrow-XL, full-XL, each alone |
| 66011763 | L40S | bf16 backbone | full-XS, full-S, each alone |
| 66003331 | L40S | bf16 attention | narrow-B, full-B, narrow-XL, full-XL, one process |

`--gres=gpu:a100` lands on either A100, and the page keeps the two apart.

## How to rebuild

The page needs only the standard library and runs on a login node:

    uv run --frozen python reports/benchmarks/training-overhead/scripts/build_page.py

It reads `data/` and the trainer's defaults, inlines everything, and prints one line per series ×
config with every number the takeaways quote.

To remeasure: `scripts/submit_overhead.sh` (it skips records already in its data file).

## Data

- `data/{a100,l40s}-bf16{backbone,attention}.jsonl`: one line per trainer-loop config, with its
  restart parts, checkpoint sizes, loop timings, and `oom` ("stage: message"). Each record names
  its chip, so the A100 files hold both A100 cards. The `-alone` files hold the one-per-process
  reruns.

The paper reads `scripts/overhead.py` and `data/` (series "A100 80 GB · bf16 backbone") in
`paper/scripts/numbers_training.py` and `fig_overhead.py`.
