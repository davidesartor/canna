# Robustness: how does the flow behave on windows it was not trained on? (2026-10-24)

The only run in `data/` is `dusty-otter-narrow-XS`, a 2,000-step narrow-XS **pipeline test**
stopped at epoch 4. It shows that every case runs end to end. Its posteriors are close to the
prior, so it says nothing about how a trained flow handles misspecification. The narrow-B and
full-B runs land around 2026-10-25; until they are probed, this report has no findings.

## Takeaways

- **Question.** The flow is trained on one noise PSD, a fixed prior and exactly `n_sources`
  sources per window. Real data break each of these. Does the flow widen gracefully, or does it
  return a confident wrong answer?
- **Method.** The flow alone, on the eval's 10 injections plus, optionally, the scorecard's
  random ones. There are nine cases:
  - the baseline;
  - the noise PSD ×0.5 and ×2;
  - the loudest source's f0 past the window's drawable band;
  - its ḟ 5σ above the prior's mean at its f0;
  - its SNR 1.5× the prior's 99.9% source SNR;
  - the quietest source removed;
  - a fifth source added;
  - an empty window.

  Each case keeps the baseline's window, truth and noise key and changes one thing, so every
  source pairs with its in-distribution self. The page shows, per case:
  - each source's width over its baseline width;
  - its offset in posterior widths;
  - 68% and 95% coverage, and a rank histogram;
  - what the flow's slots claim when there is nothing to find;
  - how often a fifth source gets a slot.
- **Answer.** Pending the trained runs. Once the runs' directories are in `outputs/runs/`, from
  the repo root:

  ```bash
  source slurm/unity.env
  sbatch $CANNA_SBATCH_GPU --partition=gpu,gpu-preempt --gres=gpu:1 reports/posterior-eval/robustness/scripts/robustness.sbatch <narrow-B run> --extra 40
  sbatch $CANNA_SBATCH_GPU --partition=gpu,gpu-preempt --gres=gpu:1 reports/posterior-eval/robustness/scripts/robustness.sbatch <full-B run> --extra 40
  ```

  Each writes `data/<run>.json` and rebuilds the page. The probe samples with the eval's 32
  steps. If [sampler convergence](../sampler-convergence/README.md) finds that 32 steps collapse
  narrow posteriors, add `--ode-steps <converged>` so the baseline itself is not too narrow.

## What the page shows

[`robustness.html`](robustness.html) is standalone and works from `file://`. The selected run,
coordinate and case are kept in the URL hash. A run with fewer than 100k training steps is badged
"untrained" and the page carries a banner saying it is a pipeline test.

- **Tiles.** The run, the windows per case, the sampler, and the check that the page's general
  matcher pairs baseline draws exactly as the eval's `match_sources` does.
- **Do the posteriors widen or shift off-distribution?** One row per case, as a median with an
  interquartile bar, and the perturbed source apart from the untouched ones. On the left is the
  width over the same source's baseline width. On the right is |median − truth| / width, against
  0.67, the median |z| of a calibrated Gaussian.
- **Is the truth still inside the flow's credible intervals?** The 68% and 95% coverage per case
  against nominal, and the selected case's rank histogram.
- **What does the flow claim where there is nothing to find?** Every slot of every window in the
  empty-window, one-fewer or noise cases, with the baseline alongside. It plots the slot's median
  claimed SNR against its f0 spread as a fraction of the window's band (0.37 is uniform over the
  band). It also marks the prior's SNR range.
- **Does the flow find a fifth source it has no slot for?** Per true source in the one-more case,
  the fraction of draws that give it a slot, against its SNR.
- **What this means.** How to read each card. For an untrained run, an explicit note that its
  numbers are not findings.

## Method

- `scripts/robustness_probe.py` runs on the run's commit with `load_trained`. The injections are
  the eval's 10 SNR quantiles and, with `--extra n`, the scorecard's first n random candidates,
  with the eval's and scorecard's noise keys. Base draws are the eval's (`sample_flow` on
  `jr.split(key, N_POSTERIOR)`), shared by every case.
- Each case is built with the repo's own simulator. `src/` is not modified.
  - **Noise cases.** `dataclasses.replace(problem, oms_noise=…·√k, acceleration_noise=…·√k)`
    simulates the observation. `preprocess` still whitens by the trained problem's PSD, as a
    deployed flow would.
  - **Prior edges.** f0 is set a quarter of the guard band (`response_points // 8` bins) past
    `f0_window`'s top. log10 ḟ is set 5 `FDOT_STD` above `log10_fdot_mean(f0)`. The SNR case
    rescales the loudest source's amplitude to 1.5× the 99.9% quantile of the per-source SNR
    over the eval's `N_CANDIDATES` prior draws (`snr_top`). The prior's amplitude has no lower
    bound, so there is no case below it.
  - **Source count.** The quietest source's amplitude is set to 0 for one fewer. One more appends
    one slot of another `sample_physical` draw for the same window. An empty window zeros every
    amplitude.
- The posterior comes from `sample_posterior` at the run's `ode_steps` (or `--ode-steps`).
  Draws are matched to the present true sources by the cheapest injective pairing in
  `match_sources`' own features. With five truths and four slots, a truth goes unpaired in a
  draw. In the baseline, this matcher must agree with `match_sources` (`matcher_mismatch`).
- Per matched source and column (f0 in bins, then the flow coordinates past f0), the probe
  records:
  - width: 1.4826 × MAD
  - offset: median − truth
  - rank of the truth
  - the fraction of draws that paired a slot to it
  - its SNR under the case's noise

  Per slot of every window, it records the f0 spread of the draws and the median SNR they would
  have against the trained PSD.

## How to rebuild

From the repo root, after `source slurm/unity.env`:

```bash
# a trained B run: 50 windows × 9 cases at 256 draws
sbatch $CANNA_SBATCH_GPU --partition=gpu,gpu-preempt --gres=gpu:1 reports/posterior-eval/robustness/scripts/robustness.sbatch <run> --extra 40
# the smoke test this folder holds (CPU, 4 cores, under 1 h for an XS run)
sbatch $CANNA_SBATCH_CPU reports/posterior-eval/robustness/scripts/robustness.sbatch dusty-otter-narrow-XS --draws 64
# the page alone (stdlib, fine on a login node)
uv run --frozen python reports/posterior-eval/robustness/scripts/build_page.py
```

The sbatch clears stray `LISA_*`, runs the probe on the run's commit via `slurm/run-src.sh`, and
then rebuilds the page. Words after the run go to the probe: `--draws`, `--extra`,
`--ode-steps`, `--cases`.

## Data

`data/<run>.json`, one per run, is rewritten after every case. It holds:

- the run's metadata: config, commit, epoch, total steps, weights, bf16, device, ODE steps, time
  power, draws, `n_sources`, `snr_top`;
- `cases`: each case's name and what it changes;
- `injections`: index, eval or random, window frequency, window SNR, and the drawable f0 band in
  bins;
- `sources`: per case and present true source, its slot and baseline slot, whether the case
  perturbed it, SNR, found fraction, and the `width`, `offset` and `rank` per column;
- `slots`: per case, window and flow slot, the f0 width and the median claimed SNR;
- `matcher_mismatch`.
