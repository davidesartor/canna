# F66 (2026-11-03): the source-count runs are launched on TREX at 90735b9, sized from 30-minute smoke tests; 16 snapshots each, one GPU per band in source-count order, forks and evals on the third

**Outcome.**
- All six (config, `n_sources`) pairs passed the handoff's smoke test on TREX at 90735b9:
  - the step benchmark;
  - a 30-minute real training job that requeued itself at 25 min and resumed from its
    checkpoint (narrow once each; full 5 times, left unattended);
  - a fork from snapshot 3 that cooled down for one epoch and finished;
  - `run.html` built.
- Epoch time (1000 steps, catalog diagnostics and checkpoint included) grows with the source
  count: 65 / 69 / 81 s for narrow-2/4/8 and 72 / 86 / 108 s for full-4/8/16.
- TREX allows 3 GPUs per user (QOS `gpu_all`), so the six runs can't run at once. Following
  Davide, each band runs on one GPU, in order of source count:
  - narrow-2 → narrow-4 → narrow-8;
  - full-4 → full-8 → full-16.
  Forks and evals share the third GPU. Each run stops at 16 snapshots (64 epochs) instead of
  32, so it can be extended later.

**Smoke tests** (jobs 16551982–93, forks 16560985, 16561821, 16561973, 16569831–33).

| pair | bench step | bench peak (GPU) | epoch time | host RSS (training) |
|---|---|---|---|---|
| narrow-2 | 65.5 ms | 2.30 GiB | 65 s | 11.6 GB |
| narrow-4 | 69.9 ms | 2.35 GiB | 69 s | |
| narrow-8 | 81.6 ms | 2.53 GiB | 81 s | |
| full-4 | 72.9 ms | 2.50 GiB | 72 s | 9.7 GB |
| full-8 | 85.7 ms | 2.61 GiB | 86 s | |
| full-16 | 107.2 ms | 2.98 GiB | 108 s | 9.3 GB |

- Both bands use the same 132.8M-parameter network. Compile takes 333–383 s, so a job's first
  epoch takes 515–600 s.
- A snapshot is ~0.68 GB.
- Host RSS reaches 11.6 GB against the jobs' default 16 GB, and the frame-only full-B run of
  [F65](065-frame-alone-diverges-sky-frame-per-window.md) was killed at 17 GB, so the runs ask for
  32 GB.
- The smoke runs' configs, histories and logs are kept locally (`outputs/runs/smoke-2026-11-03`,
  `outputs/trex-logs/smoke-2026-11-03`); their checkpoints were deleted.

**Sizing** (handoff step 2 with 16 snapshots): log_interval = 1000 · round(S · 3600 / 4 / E),
with S = 2 h (narrow) or 4 h (full); total = 64 · log_interval; warmup = total / 50; a snapshot
every 4 epochs.

| run | job | log_interval | total_steps | warmup | epoch | length |
|---|---|---|---|---|---|---|
| `261103-n2-narrow-B` | 16580353 | 28 000 | 1 792 000 | 35 840 | 30.3 min | 32.4 h |
| `261103-n4-narrow-B` | 16580354 (after n2) | 26 000 | 1 664 000 | 33 280 | 29.9 min | 31.9 h |
| `261103-n8-narrow-B` | 16580355 (after n4) | 22 000 | 1 408 000 | 28 160 | 29.7 min | 31.7 h |
| `261103-n4-full-B` | 16580356 | 50 000 | 3 200 000 | 64 000 | 60.0 min | 64.0 h |
| `261103-n8-full-B` | 16580357 (after n4) | 42 000 | 2 688 000 | 53 760 | 60.2 min | 64.2 h |
| `261103-n16-full-B` | 16580358 (after n8) | 33 000 | 2 112 000 | 42 240 | 59.4 min | 63.4 h |

All runs use the branch defaults: Muon + AdamW at lr 10⁻⁴, decay 0.1, batch 1024, linear
warmup, constant lr, then a 1 − √ cooldown over the last 20%. The chains use `afterany`
dependencies, so a failed run doesn't block the next one. Each daily requeue costs a compile
(~6 min) plus the partial epoch.

**Forks and evals: a manager job** (`canna-manager` 16580365, on `cpu2022_max`, 1 CPU, 14-day
limit). Every 10 min it keeps at most one fork or eval of its own in the queue:
- **Forks** from snapshots 1, 2, 4 and 8, i.e. epochs 4, 8, 16 and 32, as each appears.
  Snapshot 16 is the run's end, inside its cooldown. A fork from epoch 4k cools down for k
  epochs.
- **Evals** (`lisa-post.sbatch <run> eval LISA_WEIGHTS=best`) of every finished run and fork,
  ahead of forks.
- **Health check before each fork:** it skips the fork, and logs why, if the parent has a
  non-finite loss by that epoch or its catalog frame + flow NLL is more than 0.5 above its best
  so far.
- It also releases requeue holds. Its log is `outputs/manager/manager.log` on scratch.
- `sbatch` from a compute node was tested first.

**Expected timeline** (UTC, with requeue overhead):

| run | expected end |
|---|---|
| narrow-2 | ~5 Nov 00:00 |
| narrow-4 | ~6 Nov 08:00 |
| narrow-8 | ~7 Nov 17:00 |
| full-4 | ~6 Nov 08:00 |
| full-8 | ~9 Nov 02:00 |
| full-16 | ~11 Nov 20:00 |

Forks add 7.5 h per narrow run and 15 h per full run on the third GPU. The last fork (full-16
from epoch 32, 8 h) and the evals finish ~12 Nov.

**Storage.** Per main run: 16 snapshots + latest + best ≈ 12 GB. Per fork: ~1.4 GB. All six
with their forks come to ~110 GB on scratch, under the 250 GB that `infoscratch` allows for
more than a few weeks (quota 2 TB), so nothing has to leave TREX mid-run.

**Pinned code.** A run pins the checkout's commit when it starts, and narrow-8 and full-16
start days from now. `~/canna-oneshot` on TREX must therefore stay at 90735b9 until full-16 has
started (~9 Nov).
