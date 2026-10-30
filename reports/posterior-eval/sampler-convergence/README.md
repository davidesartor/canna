# Sampler convergence: when do the flow's widths stop depending on the integrator? (2026-10-24)

Probed on `misty-meadow-narrow-S` and `lucky-falcon-full-B` (2026-10-24, research log
[F60](../../research-log/060-sampler-grid-not-the-f0-floor.md)); `dusty-otter-narrow-XS` is the
2,000-step pipeline test.

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
- **Answer.** The eval's 32 uniform steps are converged, and grids that put more steps late in
  the path widen f0 instead.
  - Uniform midpoint, 32 → 256 steps: full-B 0.46 → 0.40 bins, narrow-S 0.025 → 0.028 (median f0
    width over the 40 eval sources).
  - Power-3 and log grids at 256 steps: full-B 0.62–0.65 bins, narrow-S 0.056–0.063, wider in
    every SNR band, with a worse CRPS. The network's x1 is worse late in the path than at the
    eval's last query (1 − s = 1/64), so the grid does not set full-B's f0 floor.
  - Both runs were also probed with `--schedules uniform,power2,power3,log`. To rerun, from the
    repo root:

  ```bash
  source slurm/<cluster>.env
  sbatch $CANNA_SBATCH_GPU --gres=gpu:1 reports/posterior-eval/sampler-convergence/scripts/convergence.sbatch <run> --schedules uniform,power2,power3,log
  ```

  Each writes `data/<run>.json` and rebuilds the page. Read the result off the first card: the
  step count where the uniform midpoint line reaches 1, and the f0 tile at the eval's steps.

## What the page shows

[`sampler-convergence.html`](sampler-convergence.html) is standalone and works from `file://`.
The selected run, coordinate and cell are kept in the URL hash. A run with fewer than 100k
training steps is badged "untrained" and the page carries a banner saying it is a pipeline test.

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
- **What this means.** How to read the curves, and this run's eval-steps ratio. For an untrained
  run, an explicit note that its numbers are not findings.

## Method

- `scripts/convergence_probe.py` loads the run with `load_trained` on the run's commit. It
  rebuilds the eval's injections (`eval_injections(problem, config.seed)`), its observations
  (`sample_observation` with the eval's noise keys) and its base draws (`sample_flow` on the
  first `--draws` of `jr.split(key, N_POSTERIOR)`), so the eval's 32-step cell is the eval's own
  posterior.
- `sample_on_grid` is the eval's sampler on an arbitrary grid of times. It moves `x_t` along the
  velocity (m − x_t)/(1 − t), m the flow's mean of x₁: midpoint makes two calls a step, Euler one.
  The `log` schedule spaces 1 − t log-uniformly from 1 down to 10^−`LOSS_TIME_DECADES`, then
  takes the last step to t = 1. The eval's own
  `sample_posterior` runs too, and the grid sampler must match it on the uniform grid at the
  eval's step count (`eval_match`).
- The draws are matched to the true sources with `match_sources` and mapped to columns: f0 in
  frequency bins, then the seven flow coordinates past f0 (log ḟ, l, b, Re/Im z±). Per source and
  column the probe records:
  - width: 1.4826 × MAD
  - offset: median − truth
  - rank of the truth among the draws
  - sample CRPS at the truth

## How to rebuild

From the repo root, after `source slurm/unity.env`:

```bash
# a trained B run: the full grid (4…256 steps, 256 draws), ~1.5k network calls per draw per schedule
sbatch $CANNA_SBATCH_GPU --partition=gpu,gpu-preempt --gres=gpu:1 reports/posterior-eval/sampler-convergence/scripts/convergence.sbatch <run>
# the smoke test this folder holds (CPU, 4 cores, under 1 h for an XS run)
sbatch $CANNA_SBATCH_CPU reports/posterior-eval/sampler-convergence/scripts/convergence.sbatch dusty-otter-narrow-XS --draws 32 --steps 4,8,16,32,64,128
# the page alone (stdlib, fine on a login node)
uv run --frozen python reports/posterior-eval/sampler-convergence/scripts/build_page.py
```

The sbatch clears stray `LISA_*`, runs the probe on the run's commit via `slurm/run-src.sh`, and
then rebuilds the page. Words after the run go to the probe: `--draws` (at most 256),
`--steps`, `--schedules` (midpoint), `--euler`.

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
