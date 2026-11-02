# Robustness: how does the flow behave on windows it was not trained on? (2026-11-02)

Run: `261101-royal-ferret-narrow-B` (epoch 1250, bf16 backbone, 128 draws, 32 ODE steps), 32 injections
per case (128 sources each; the page's windows tile reads the same). `dusty-island-full-B` was not probed.

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
- **Answer.** Mostly it widens only a little and does not go confidently wrong, with weak spots
  in calibration. Baseline f0 68% coverage is 0.60 (width ratio 1 by definition), noise x0.5
  keeps it at 0.61, noise x2 drops it to 0.44 with f0 widths 0.80x the baseline's (median
  |offset|/width 1.21 against 0.79). f0 past the edge gives 0.42 (|offset|/width 1.26), fdot
  above the prior 0.63, SNR above the prior 0.53 (widths 0.95x). One source fewer widens nothing
  (0.89x, coverage 0.75); one source more narrows the others to 0.74x (coverage 0.52), and the
  extra source gets a slot in 0.89 of draws (median found fraction). In the empty window the 128
  slots claim a median SNR of 0.34, the same as the baseline's, with f0 spread 29.1 against 27.0
  bins. Even the baseline sits at 0.60, below 0.68, so the cases are read against it.

  The sources in the baseline have median |offset|/width 0.79 (0.67 is calibrated).

  Rebuild, from the repo root (single bare sbatch; `PYTHON` is the project venv's python):

  ```bash
  sbatch --export=ALL,PYTHON=$PWD/.venv/bin/python --partition=gpu-highvram,gpu-preempt-highvram --account=pi_mehuber_umass_edu --qos=normal --constraint=a100 --gres=gpu:1 reports/posterior-eval/robustness/scripts/robustness.sbatch 261101-royal-ferret-narrow-B --draws 128 --injections 32
  ```

  The probe samples with the run's 32 steps. If [sampler convergence](../sampler-convergence/README.md)
  finds that 32 steps collapse narrow posteriors, add `--ode-steps <converged>`.

## What the page shows

[`robustness.html`](robustness.html) is standalone and works from `file://`. The selected run,
coordinate and case are kept in the URL hash. A run with fewer than 100k training steps is badged
"untrained".

- **Tiles.** The run, the windows per case, the sampler, and the check that no baseline source's
  f0 match is off its slot index (0% expected).
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
- **What this means.** How to read each card. 

## Method

- `scripts/robustness_probe.py` runs on the working tree's `src` with `load_trained`. The injections are
  the eval's first `--injections` (default 128, all sorted by f0), with the eval's noise keys.
  Base draws are the eval's (`jr.normal(fold_in(key, 1), ...)`), shared by every case.
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
  Draws are matched to the present true sources by f0 order (the flow's slots are f0-ordered); with
  five truths and four slots, a truth goes unpaired in a draw. In the baseline, the match must be
  the slot index (`matcher_mismatch`, 0.0 here).
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

From the repo root, as the single bare sbatch above (GPU, about 9 min for 32 injections and
128 draws on an A100). The page alone (stdlib, fine on a login node):

```bash
python3 reports/posterior-eval/robustness/scripts/build_page.py
```

The sbatch clears stray `LISA_*`, takes bf16 from the run's config, runs the probe on the working
tree's `src`, and then rebuilds the page. Words after the run go to the probe: `--draws`,
`--injections`, `--ode-steps`, `--cases`.

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
