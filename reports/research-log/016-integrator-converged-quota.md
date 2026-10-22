# F16 (2026-10-03): the integrator is converged, and E3' died on the home-directory quota
- 64 and 128 RK4 steps match 32 within noise: XS-late's detection loss and 0.088-bin floor
  belong to the network, not the eval's ODE. 32 steps stay the default.
- E3' (XS-late-768) died at its first checkpoint on the per-user quota.

**T2** (jobs 13641220, 13641221; 200 random injections):

| steps | found, SNR < 15 / 15–40 / 40–100 / ≥ 100 | f₀ width (bins) | rank |
|---|---|---|---|
| 32 | 0.46 / 0.81 / 0.92 / 0.99 | 0.152 / 0.129 / 0.087 / 0.088 | 0.51 / 0.46 / 0.46 / 0.50 |
| 64 | 0.46 / 0.81 / 0.92 / 0.99 | 0.147 / 0.130 / 0.088 / 0.089 | 0.50 / 0.46 / 0.46 / 0.50 |
| 128 | 0.47 / 0.81 / 0.92 / 0.99 | 0.147 / 0.127 / 0.087 / 0.089 | 0.50 / 0.46 / 0.46 / 0.50 |

**E3' failure.** Job 13641222 trained epoch 1 (129.5 s incl. compile), then orbax failed with
`RESOURCE_EXHAUSTED` writing `outputs/lisa-XS-late-768/checkpoints/1.orbax-checkpoint-tmp`
(exit 1).
- The filesystem has 5.8 TB free, so it is the **per-user quota on /home/mp/mentasgi**. A
  768-wide checkpoint is ~1.7 GB (0.77 GB at XS), and orbax holds two while saving.
- `outputs/` holds eight checkpoint directories, four obsolete: `lisa-XS.nan` and `lisa-S.nan`
  (NaN weights), `aux10` (stopped E1), `lisa-B` (empty); plus the failed temporary checkpoint.
- `500k/lisa-XS` and `lisa-XS` (1M) are backed up on the laptop. There is no personal folder
  under `/work/LISA`, where other LISA users keep theirs.
