# Learned loss weights: what the per-bin, per-group `loss_log_var` learns, and what it ends up weighting (2026-10-24)

## Takeaways

- **The only checkpoint on Unity is a 2000-step smoke run (dusty-otter-narrow-XS), so the learned
  weights here are not evidence yet.** The page and the probe are ready for narrow-B and full-B.
- **28.2% of training draws are later than the default sampler ever queries.**
  - Training mixes uniform t with log-uniform 1 − s on [1e-4, 1] at `late_time_mix` 0.5.
  - The 32-step midpoint sampler's last query is at 1 − s = 1/64 (log10 = −1.81).
  - So the bins below −1.81 (six full non-empty bins from −4.125 to −1.875, plus a sliver of the
    next) hold 28.2% of the draws, and the network's training there never serves a default draw.
  - This needs no network: it follows from the config. It also means the five bins below
    log10(1 − s) = −4.125 get no draws at all and keep their init λ.
- **λ is on AdamW, so it moves at most about its learning rate per step.**
  - In dusty-otter the largest move from init tracks that bound: 0.34, 0.64, 0.96 and 1.28 after
    500, 1000, 1500 and 2000 steps, against bounds of 0.4, 0.9, 1.4 and 1.9. That is about 2/3 of
    the bound after warmup, so λ was still travelling at the end, not settled.
  - Over a real run the bound is 650 (narrow) or 240 (full) nats, so it stops mattering. Whether λ
    has reached log err there is what the page's third card checks.
- **Per-bin training density** (share of draws per log10(1 − s) bin from −6 to 0, 0.375 decades
  each): 0 for the five bins below −4.125, then 0.031, 0.047, 0.047, 0.048, 0.049, 0.051, 0.056,
  0.069, 0.098, 0.169 and 0.336. Late bins get ~5% each from the log-uniform half, and the
  earliest bin takes a third of everything.

## What the page shows

[learned-loss-weights.html](learned-loss-weights.html). The top bar picks a run. Runs under 100k
steps show a smoke banner.

- **Tiles:** the run, its trained steps, the largest λ move against the AdamW bound, and the share
  of draws past the sampler.
- **Where training draws land:** the density per time bin, with the default sampler's last query
  as a vertical line.
- **λ over training:** for one coordinate group (f0, Mc, sky, z+, z−), λ per bin at each saved
  epoch, with the init dashed and the log of the measured bin error dotted.
- **Has λ caught up with the error?** λ − log err per bin and group at the last epoch. At
  equilibrium this is 0.
- **What the weights do at the end:** each (bin, group)'s share of the error term under the learned
  weights against the plain flow loss. A table gives the per-group shares and the geometric-mean
  per-coordinate weight at init and at the end.
- **What this means.**

## Method

`scripts/loss_weights_probe.py` runs on the run's own commit (via `slurm/run-src.sh`):

- λ at init comes from `setup(cfg)` (the run's own seed). λ at the end comes from the last
  checkpoint, for both the averaged weights x and the training weights y.
- The history's last λ is checked against the checkpoint's y (dusty-otter: identical).
- λ over training and the per-bin error come from `history.npz` (`log_var`, `bin_error`).
- **Per-coordinate weight**, relative to the plain loss's mean over the 8 coordinates:
  e^{−λ} · 8 / (5 |g|).
- **Error share:** density × err × e^{−λ} (learned) against density × err × |g| / 8 (plain).
- **Density per bin:** computed in closed form from `late_time_mix`, `late_time_floor` and
  `time_power`. Draws below the first edge are clipped into the first bin, as the code does.
- **AdamW bound:** lr · steps² / (2 · warmup) during warmup, then lr · (steps − warmup / 2).

## Rebuild

From the repo root, CPU (enough for any size; it reads two checkpoints and runs no forward pass):

    source slurm/unity.env; sbatch $CANNA_SBATCH_CPU reports/architecture/learned-loss-weights/scripts/probe.sbatch <run>

The job writes `data/<run>.json` and rebuilds the page from every JSON in `data/`. To rebuild
only the page (stdlib only): `uv run --frozen python reports/architecture/learned-loss-weights/scripts/build_page.py`.

## Data

`data/<run>.json`, one per run.
- **Config:** run, commit, epoch, device, steps.
- **Time bins:** `time_edges`, `density`, `sampler_min_decade`, `unqueried_share`.
- **λ:** `log_var` [epoch, bin, group]; `init`, `final_average` and `final_training` [bin, group].
- **Error:** `bin_error`, stored as log err.
- **Movement:** `moved` and `adam_reach` per epoch.
- **Weights and shares:** `weight_init`, `weight_final`, `learned_share`, `plain_share`.

Present: `dusty-otter-narrow-XS` (smoke, 2000 steps, commit 11fc8a4, CPU).
