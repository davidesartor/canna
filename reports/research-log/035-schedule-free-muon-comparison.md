# F35 (2026-10-13): schedule-free Muon comparison, stopped at 9k steps — weight decay slows the growth; z drifts from x regardless

- Decay 0.05 slows weight growth ~40%, at no loss cost so far.
- z drifts from x at the same rate with or without decay.
- Early on, the average x is the better model.

**Runs.** Four 150k-step XS variants (`slurm/lisa-compare.sbatch`; 16 bins per token, ordered
sources, reconstruction head; b9bb3ef, 9bdc2d4), cancelled on 13 Oct for lack of GPU time. A
(14408555, schedule-free, Muon momentum 0.95, decay 1e-6, XS-fd's settings) and B (14408558,
decay 0.05) ran 9k steps (warmup ending at 3k), 31–32 s per epoch. C (no Muon momentum) never
finished an epoch; D (no schedule-free) never started. No scorecards: the checkpoints were on
node /tmp.

**Diagnostics** (epoch 9; growth over epochs 5–9):

| | A (decay 1e-6) | B (decay 0.05) |
|---|---|---|
| held-out loss, x | 0.9660 | 0.9655 |
| held-out loss, y | 0.9684 | 0.9663 |
| backbone weight RMS (initial 0.0192) | 0.0287 (+50%) | 0.0251 (+31%) |
| growth per 1k steps | 0.0013, steady | 0.0008, slowing |
| z − x RMS | 0.0214 | 0.0203 |

**Readings (9k steps, one seed).**
- Decay 0.05 does what F34 predicted. A grows at an undiminished rate, toward XS-fd's 19×.
- z drifts from x at ~0.002 RMS per 1k steps either way, already 0.7–0.8 of the weights' RMS.
  Decay does not hold z near its average; only the lr or the averaging can.
- x's held-out loss is 0.001–0.002 below y's from epoch 4, as schedule-free intends. XS-fd's
  reversal (x worse by 0.012) is late-training, never reached here.
- Untested: whether Muon's momentum hurts inside schedule-free (C), and whether schedule-free
  beats constant-lr Muon at all (D).

**Next.** Run with decay ~0.05; keep logging x and y, weight RMS and z − x (now default), and
score both x and y. The late-phase lag needs a full-length run.
