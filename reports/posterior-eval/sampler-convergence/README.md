# Sampler convergence: when do the flow's widths stop depending on the integrator? (2026-11-02)

Probed on `261101-royal-ferret-narrow-B` (epoch 1250, bf16 backbone; job 66528675, 2026-11-02).
The older runs from before f0 ordering are in research log
[F60](../../research-log/060-sampler-grid-not-the-f0-floor.md). `dusty-island-full-B` was not probed.

## Takeaways

- **Question.** The eval samples every posterior with 32 uniform midpoint steps (`ODE_STEPS`, 64
  network calls per draw). Narrow posteriors are resolved only late in the
  path, so a coarse grid can collapse them: the draws come out narrower than the flow's own
  posterior. At which step count do per-source widths stop changing, and are the eval's 32
  steps past it?
- **Method.** One trained flow and the eval's own 10 injections, observations and base draws.
  Each is sampled at 4 to 256 steps on three step schedules (uniform, power 2, log-spaced in
  1 − t down to the loss's 1e-6 time floor), with the eval's midpoint rule and with Euler. Every
  schedule feeds the network the time it was trained on, so only where the steps fall changes.
  Per source and coordinate (f0 first), the probe records the width (1.4826 × MAD), the offset
  and rank of the truth, and the CRPS. The page shows each as a ratio to the finest log-schedule
  cell.
- **Answer.** On the narrow-B run the eval's 32 uniform steps are converged, to the reference
  (log schedule, 128 steps, 64 draws, 512 sources).
  - Uniform midpoint, f0 width over the reference: 0.971 at 4 steps, 0.994 at 32, 0.999 at 128.
    The other coordinates are within 4% at 32 (0.97-1.02). Median f0 width at 32 steps is 27.0
    bins against 28.1 for the reference.
  - Power-2 and log grids reach 1.00 by 32 steps; the log grid is the least collapsed at 4 steps.
    Midpoint and Euler 128 steps give 0.985 and 1.00 for f0 (Euler at 32: 0.958).
  - Median f0 CRPS ratio is 0.99 everywhere, so more late steps do not change the score.
  - So no collapse for this flow at the eval's steps; the earlier late-time floor story
    belongs to the old runs. The `power3` schedule and 256 steps were not run.
  - The grid sampler matches `sample_posterior` on the eval's grid to max |Δx| 0.055 (bf16
    backbone, different jit shapes), against f0 widths of 27 bins.

  To rerun, from the repo root (single bare sbatch):

  ```bash
  sbatch --export=ALL,PYTHON=$PWD/.venv/bin/python --partition=gpu-highvram,gpu-preempt-highvram --account=pi_mehuber_umass_edu --qos=normal --constraint=a100 --gres=gpu:1 reports/posterior-eval/sampler-convergence/scripts/convergence.sbatch 261101-royal-ferret-narrow-B --draws 64 --steps 4,8,16,32,64,128
  ```

  It writes `data/<run>.json` and rebuilds the page. Read the result off the first card: the
  step count where the uniform midpoint line reaches 1, and the f0 tile at the eval's steps.

## What the page shows

[`sampler-convergence.html`](sampler-convergence.html) is standalone and works from `file://`.
The selected run, coordinate and cell are kept in the URL hash. A run with fewer than 100k
training steps is badged "untrained".

- **Tiles.** The run, the eval's sampler, the selected coordinate's median width at the eval's
  steps over the reference, and the check that the probe's grid sampler reproduces
  `sample_posterior` on the eval's grid (max |Δx|).
- **At which step count do the widths stop changing?** The median over sources of width over
  reference width against step count. There is one line per schedule (colour) and rule (solid
  midpoint, dashed Euler), with an interquartile band on the selected line. The eval's step count
  is marked.
- **Does the default step count collapse the narrow posteriors?** Per source, the selected
  cell's width over the reference width, against source SNR. A collapse that worsens with SNR
  tilts the cloud down to the right.
- **Does the score keep improving with more network calls?** The median CRPS over the reference
  CRPS against network calls per draw. Cost per call is in
  [sampling cost](../../benchmarks/sampling-cost/sampling-cost.html).
- **What this means.** How to read the curves, and this run's eval-steps ratio.

## Method

- `scripts/convergence_probe.py` loads the run with `load_trained` on the working tree's `src`. It
  rebuilds the eval's injections (`eval_injections(problem, config.seed)`, f0-sorted), its
  observations (`sample_observation` with the eval's noise keys) and its base draws (the first
  `--draws` of `jr.normal(fold_in(key, 1), (N_POSTERIOR, ...))`), so the eval's 32-step cell is the eval's own
  posterior.
- `sample_on_grid` is the eval's sampler on an arbitrary grid of times. It moves `z_t` along the
  velocity (m − z_t)/(1 − t), m the flow's mean of z₁ in the reference frame (x₁ = μ + σ z₁): midpoint makes two calls a step, Euler one.
  The `log` schedule spaces 1 − t log-uniformly from 1 down to 10^−`LOSS_TIME_DECADES`, then
  takes the last step to t = 1. The eval's own
  `sample_posterior` runs too, and the grid sampler must match it on the uniform grid at the
  eval's step count (`eval_match`).
- Draws and truths are both f0-sorted, so slot i is source i; columns are: f0 in
  frequency bins, then the seven flow coordinates past f0 (log ḟ, l, b, Re/Im z±). Per source and
  column the probe records:
  - width: 1.4826 × MAD
  - offset: median − truth
  - rank of the truth among the draws
  - sample CRPS at the truth

## How to rebuild

From the repo root, as the single bare sbatch above (GPU, ~17 min on an A100 for this grid).
The page alone (stdlib, fine on a login node):

```bash
python3 reports/posterior-eval/sampler-convergence/scripts/build_page.py
```

The sbatch clears stray `LISA_*`, takes bf16 from the run's config, runs the probe on the working
tree's `src`, and then rebuilds the page. Words after the run go to the probe: `--draws` (at most
256), `--steps`, `--schedules` (midpoint), `--euler`.

## Data

`data/<run>.json`, one per run, is rewritten after every cell, so a job cut short still leaves a
record. It holds:

- the run's metadata: config, commit, epoch, total steps, weights, bf16, device, prediction,
  self-conditioning, time power, the eval's step count, the log floor, draws and steps;
- `sources`: per source, the injection, slot, source and window SNR, and f0 in bins;
- `eval_cell`: the repo's `sample_posterior` at the eval's step count;
- `cells`: per rule, schedule and step count, the network calls, seconds, and the `width`,
  `offset`, `rank` and `crps` arrays [source][column];
- `eval_match`: the grid sampler against `sample_posterior` on the eval's grid.
