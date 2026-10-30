# Data dependence: how much each source slot's velocity follows the data (2026-10-24)

## Takeaways

- **No trained run has been measured yet.** The only record is the 2000-step smoke run
  `dusty-otter-narrow-XS` (0.3% of narrow's 650k steps). It checks the pipeline end to end and
  is labelled as a smoke run on the page. Its numbers say nothing about the architecture.
- The report will answer three questions once narrow-B and full-B have trained:
  - **Does the velocity read the data, or only x_t?** For each coordinate, the data share and
    the x_t share of the velocity's variance, against path time t and SNR. A data share that
    collapses before the posterior width is reached would mean the flow stops reading the data
    too early. Compare it with the late-time widths in
    [reference comparison](../../posterior-eval/reference-comparison/).
  - **Does each slot read its own source's band?** The share of the squared Jacobian on the bins
    holding 95% of the slot's own source's power, against the share uniform sensitivity would
    give.
  - **How local is the reading?** The sensitivity per bin against the distance from the slot's
    f0.
- To fill it, run the probe on each new run (see [How to rebuild](#how-to-rebuild)). The page
  then defaults to the longest-trained non-smoke run.
- **The network reads the data's phase** (velocity probe, smoke run `dusty-otter-narrow-XS`, epoch 4,
  1,024 fresh training samples). A 0.3 rad rotation of the data moves the z± velocity by 0.36–0.39 of
  the target's spread (swapping in another sample's data moves it by 1.1–1.4). Early in the path
  (s < 0.5), the error in φ± of the x₁ the head implies is 0.32/0.36 rad. For x_t itself it is
  0.91/0.95. So the excess width of this run comes from training length or capacity, not from a
  network blind to the phases. This is a smoke-run result and is labelled so on the page. Rebuild:
  `velocity_probe.sbatch` (see [How to rebuild](#how-to-rebuild)).

## What the page shows

[data-dependence.html](data-dependence.html). The top bar picks a run (smoke runs are marked) and
a path time t. A banner flags any run under 50k steps as a smoke run.

1. **How much of the velocity moves with the data, and how much with x_t?** The data share and the
   x_t share against t, one line per coordinate, for all slots or one slot.
2. **Does a louder source's velocity depend more on the data?** The data share at the selected t
   by source SNR.
3. **Is a slot sensitive to its own source's band?** The squared Jacobian's split into own band,
   other sources' bands and rest, against t. Each set's uniform baseline is drawn dashed. The
   chart shows all coordinates or one.
4. **How far from its f0 does a slot read?** Sensitivity density in rings around the slot's f0
   bin, relative to uniform, per coordinate.
5. **Where does one observation's velocity look?** One observation's sensitivity per window bin
   and slot, with its sources' bands and f0 marked.
6. **Has the network learned each coordinate, and does it read the data's phase?** The
   [velocity probe](scripts/velocity_probe.py), one card with a metric switch: R² of the predicted
   velocity against the training target per coordinate and band of path position s; its spread and
   the output head's row norms; its response to rotating or swapping the data; and the head's φ±
   error against x_t's. A probe record is picked with the Probe switch when there is more than one.
7. **What this means**: how to read each card. It lists which runs are smoke and which are
   trained.

The URL hash `#run=<tag>` selects a run and `#probe=<tag>` a velocity-probe record.

## Method

[`scripts/data_dependence.py`](scripts/data_dependence.py) loads a run's checkpoint like
[the velocity probe](scripts/velocity_probe.py):
`config.load_run()`, `load_trained`, the run's own eval weights. It runs on the run's commit
(`slurm/run-src.sh`).

- **The velocity.** The one the sampler follows, `(m − x_t)/(1 − t)` with m the head's mean of x1
  (one network call; data/ predates this and was measured on the earlier self-conditioned network).
- **Observations.** 512 fresh training observations from `train_sample` (keys 300–307 × 64), at
  t ∈ {0.1, 0.5, 0.9, 0.99, 0.999, 0.9999}. Every t sees the same observations, with two base
  draws each.
- **Shares (total effects).** The velocity is recomputed in two ways:
  - with the data swapped for another observation's y, x_t held: `data = E(v − v_y)² / 2 Var v`;
  - with x_t swapped for the second base draw, the data held: `x_t = E(v − v_x)² / 2 Var v`.

  Both are per coordinate and are reported for all slots, per slot and per SNR bin (7–20–60–200–1000).
  1 means independent, 0 means no dependence. Interactions count in both, so the two need not add
  to 1, and either can exceed 1 when the swapped velocities anti-correlate.
- **Sensitivity.** The Jacobian of the velocity in the whitened data `y` (`jax.jacrev`, the first
  32 observations). Its square is summed over the 6 channels into a [slot, coordinate, bin] map
  and normalised per slot and coordinate. That map is split three ways:
  - the slot's **own band**: the fewest window bins holding 95% of its source's whitened power,
    `2|h̃|²/window_power` from `source_segments`. This covers the Doppler spread and the
    antenna sidebands.
  - the **other sources' bands**: their union, minus the own band.
  - **the rest**.

  The baseline is each set's bin fraction. The ring enrichment pools slots and observations over
  |bin − f0 bin| ∈ {0, 1, 2–3, 4–7, …}.
- Slots match sources by index, since sources are sorted loudest first, as in training.

**Caveat.** The data swap keeps the observation's own `f`, so the swapped y comes from another
window centre. The data share therefore includes the flow's reaction to a mismatched (y, f). That
is still "the data changed", but it is not a same-window redraw.

## How to rebuild

From the repo root on a login node, `source slurm/unity.env` first. On a new run (narrow-B or
full-B; GPU for B and up):

    sbatch $CANNA_SBATCH_GPU --partition=gpu,gpu-preempt --gres=gpu:1 reports/architecture/data-dependence/scripts/data_dependence.sbatch <run>

An XS or S run fits the CPU partition (the smoke run took 2 min of probe on 4 CPUs):

    sbatch $CANNA_SBATCH_CPU reports/architecture/data-dependence/scripts/data_dependence.sbatch <run>

The optional arguments are `[tag] [training]`. The tag names the record (default `<run>`);
`training` probes the training weights instead of the schedule-free average. The job writes
`data/<tag>.json`, then rebuilds the page. To rebuild the page alone (stdlib only):

    uv run --frozen --no-sync python reports/architecture/data-dependence/scripts/build_page.py

### Velocity probe

[`scripts/velocity_probe.py`](scripts/velocity_probe.py) measures one checkpoint on 1,024 fresh training
samples (one draw each, the velocity the sampler follows) and writes `data/velocity-probe/<tag>.json`:
the predicted velocity's spread over the target's, its R² per band of s, the φ±
error of the implied x₁ next to x_t's, the head's last-layer row norms, and the velocity's rms change under
a rotation of the whitened data by e^{iα} or a swap with another sample's data. From the repo root, after
`source slurm/unity.env`:

    sbatch $CANNA_SBATCH_CPU reports/architecture/data-dependence/scripts/velocity_probe.sbatch <run> [tag] [training]
    sbatch $CANNA_SBATCH_GPU --partition=gpu,gpu-preempt --gres=gpu:1 reports/architecture/data-dependence/scripts/velocity_probe.sbatch <run> [tag] [training]

It rebuilds the page last. For a job still training,
[`scripts/velocity_probe_live.sh`](scripts/velocity_probe_live.sh) `<run> <jobid> <tag>` probes a snapshot of
its checkpoint on the job's own GPU (rebuild the page after).

## Data

`data/<tag>.json`, one per probed run. It holds:
- the run's identity: `run`, `config`, `commit`, `epoch`, `steps`, `total_steps`, `weights`,
  `bf16`, `device` and `seconds`;
- the grids: `coordinates`, `t_grid`, `snr_edges`, `window_bins`, `patch`, `observations`,
  `jacobian_observations`, `band_power` and `rings`;
- `shares[t]`: `{all, slot[s], snr[i]}`, each `{data[8], xt[8], n}`;
- `jacobian[t]`, with:
  - `split` [slot, coord, own/other/rest];
  - `split_snr[i]` [coord, 3];
  - `enrichment` [coord, ring];
  - `example` [slot, bin / stride];
- `baseline` [slot, 3], the uniform shares;
- `example`: observation 0's `snr`, `f0_bin`, `own` mask and `stride`.

| record | run | steps | job |
|---|---|---|---|
| `dusty-otter-narrow-XS.json` | smoke, commit 11fc8a4 | 2000 | 66014207 (probe), 66014323 (end to end) |

`data/velocity-probe/<tag>.json`, one per probed checkpoint: `run`, `config`, `epoch`, `weights`, `device`,
`samples`, `coordinates`, `spread`, `head_rows`, `bands` (per s band: `n`, `r2`, `head_error`, `xt_error`)
and `changes` (per rotation and `other data`).

| record | run | steps | job |
|---|---|---|---|
| `velocity-probe/dusty-otter-narrow-XS.json` | smoke, epoch 4 | 2000 | moved from flow-vs-reference |
