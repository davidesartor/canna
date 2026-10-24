# Flow vs reference: where the flow is wrong, and why

Only one run is in `data/` so far: `dusty-otter-narrow-XS`. It is a 2,000-step narrow-XS
**pipeline test** stopped at epoch 4. Its numbers show that the comparison works from end to end.
They do not show how good the method is. The narrow-B and full-B runs drop in when the jobs under
[How to rebuild](#how-to-rebuild) are run again.

## Takeaways

- **The flow is far too wide, and the pinned phase combination shows it best.** The signal is
  linear in z₊ ∝ A(1 − sin ι)² e^{iφ₊} and z₋ ∝ A(1 + sin ι)² e^{iφ₋}, with φ± = φ₀ ± 2ψ. A
  face-on source pins φ₋ and a face-off one pins φ₊. Against the per-source MCMC (40 sources), the
  pinned φ± is a median 50× wider (3.1–1444×). φ₀ alone reads 22.5× (1.9–1163×), because it
  inherits ψ's spread. Against the joint MCMC (16 sources), pinned φ± is 36× (1.8–1455×) and φ₀
  is 5.4×.
- **f₀ is the widest parameter.** The flow is a median 837× wider than the per-source MCMC
  (53–8450×). Across the joint windows, 45–98% of the flow's draws per source lie more than a
  frequency bin from the MCMC's median f₀. log Mc is the exception at 0.99× (0.91–1.11×): its
  posterior is the prior, so it says nothing about the flow.
- **The gap grows with SNR.** The median pinned φ± ratio is 8.7× below SNR 15 (8 sources), 33×
  at 15–100 (12), 84× at 100–300 (9) and 117× above 300 (11). The same bands for φ₀ give 2.7×,
  8.8×, 46× and 157×. The correlation between log ratio and log SNR is 0.80. The MCMC narrows as
  the SNR rises, but the flow does not yet.
- **The flow is too wide, not offset.** Over the eight linear parameters, the truth falls in the
  flow's central 68% interval for 79% of sources, and in its central 95% interval for 97%. Both
  are above nominal, which is what an over-wide posterior gives. The joint MCMC's median sits in
  the flow's central 95% for every parameter of 15 of the 16 sources.
- **The per-source MCMC is a fair reference here.** The per-source MCMC holds the other sources
  at their truth, so it could be narrower than the joint posterior where sources overlap. Over
  the 16 sources both MCMCs cover, the joint width over the per-source width has a median of
  1.0×. Only 3% of entries are above 1.5×, and the largest is 3.1× (f₀ of injection 2, slot 3,
  SNR 7.25).
- **The network reads the data's phase.** A 0.3 rad rotation of the data moves the z± velocity
  by 0.36–0.39 of the target's spread (swapping in another sample's data moves it by 1.1–1.4).
  Early in the path (s < 0.5), the error in φ± of the x₁ the head implies is 0.32/0.36 rad. For
  x_t itself it is 0.91/0.95. So the excess width comes from training length or capacity, not
  from a network blind to the phases.
- **Mind the sampler.** The draws come from eval `4-average-32`: 32 uniform x-pred ODE steps with
  time_power 1.0. That sampler is known to collapse narrow posteriors. A collapse can only make a
  posterior narrower, so a flow this much wider than the reference is the network's doing. Once a
  trained run comes within a few times the reference width, evaluate it with more steps before
  reading its widths.

## What the page shows

[`flow-vs-reference.html`](flow-vs-reference.html) is standalone and works from `file://`. The
sidebar sets the run (smoke runs are badged "test"), the reference (per-source or joint MCMC),
the parameter (pinned φ± or any of the ten) and the selected source. A dot clicked in any card
selects that source in every card. The state is kept in the URL hash. Tiles above the cards give
the run, the sampler, the median pinned φ± and φ₀ ratios, truth coverage and the phase response.

1. **Which parameters is the flow too wide in?** One strip per parameter: flow width over
   reference width per source, on a log axis, with each row's median.
2. **Does the gap grow with SNR?** The selected parameter's ratio against source SNR. When
   pinned φ± is selected, the matching φ₀ ratio shows as rings.
3. **Does the truth fall where the flow says?** On the left, a histogram of the truth's rank among
   the flow's draws, with a 95% binomial band; the selected source is marked. On the right, 68%
   and 95% coverage per parameter. This card needs no MCMC, so it works for every band.
4. **What does the flow's posterior look like next to the joint MCMC?** A window and one of its
   sources: a scatter of any two parameters' offsets from the truth, with marginals. The flow is
   blue, the joint MCMC gray, and the Fisher Gaussian orange (optional). The view zooms to the
   MCMC or to all draws. A table holds the source's full comparison: widths, W1, Fisher, seed
   agreement, coverage of the MCMC median, and the fraction of draws more than a bin off in f₀.
5. **Has the network learned each coordinate, and does it read the data's phase?** The velocity
   probe, per flow coordinate. It shows one of four metrics: R² by band of path position, the
   velocity's spread and the head's row norms, the response to a rotated or swapped data window,
   or the φ± error of the head against x_t. When a run has probes at more than one epoch, you can
   pick the epoch.

A "What this means" section closes the page. It is written from the selected run's data.

## What this means

On the pipeline test, the flow is centred on the right answer but is tens to hundreds of times
too wide, and wider the louder the source. The gap is not an artefact of the reference: the
per-source and joint MCMC agree within a few percent at the median. It is not the sampler
either, since a collapse would narrow the flow, not widen it. The network does respond to the
data's phase, so the open question is whether training to the full step count closes the gap.
The narrow-B run answers that on every card. The full-band run has no MCMC, so cards 3 and 5 are
the only ones that apply to it.

## Method

- **Widths** ([`draw_widths.py`](scripts/draw_widths.py)) use the run's newest eval at its config's
  weights and ode_steps.
  - Each source's flow draws are matched to the truth and folded (`mcmc.fold`). The parameters
    are f₀ in bins, log Mc, log A, sky, ψ, sin ι and φ₀, plus φ± read off `physical_to_flow` at
    the chart's phase reference.
  - Widths are 1.4826·MAD, and circular (√(−2 ln R)) for φ±. The same widths are taken of the
    cached narrowband MCMC (40,000 draws, one ensemble per source with the others at their
    truth).
  - The truth's rank among the draws is recorded too; φ± are cut opposite the draws' circular
    mean.
  - "Pinned" is the narrower of the MCMC's φ₊ and φ₋. For ranks, it is chosen by the sign of
    sin ι.
  - φ₀ is now a linear MAD on the folded draws. The old draw-widths page used a circular width
    of φ₀ mod π, so its 7.5× median is not comparable to the 22.5× here.
- **Joint comparison** ([`compare_joint.py`](scripts/compare_joint.py)) compares the eval draws of narrow
  windows 2, 5, 7 and 9 with the joint MCMC of the
  [mcmc report](../../mcmc/) (`scripts/joint_mcmc.py`, two seeds per window).
  - Per source it gives each set's widths, W1 over the MCMC width, the Fisher Gaussian's width,
    the agreement of the two seeds, and whether the MCMC median lies in the flow's 68%/95%
    intervals. It also keeps 2,000 draws per set; the page keeps 400.
- **Velocity probe** ([`velocity_probe.py`](scripts/velocity_probe.py)) restores the checkpoint. It uses
  the averaged weights, or the training weights with `LISA_WEIGHTS=training`. It draws 16 × 64
  fresh training samples, one draw per observation, self-conditioned as the sampler runs. Per
  flow coordinate it records:
  - the velocity's spread over the target's;
  - R² per band of s (0–0.5, 0.5–0.9, 0.9–0.99, 0.99–1);
  - the head's last-layer row norms over their median;
  - the velocity's rms change when the whitened data are rotated by e^{iα} (α = 0.3, π/2, π;
    this moves only φ₀ in the true posterior) or swapped for another sample's;
  - the φ± error of the implied x₁ next to x_t's own.
- [`build_page.py`](scripts/build_page.py) reads `data/` and writes the page. The data are inlined as
  JSON, and runs are keyed by name. A run trained for fewer than 20,000 steps is flagged as a
  pipeline test.

Runs on the old cos/sin φ₀ chart are covered in research-log entries
[F47](../../research-log/047-b-template-b-fisher-no-phi0.md)–[F54](../../research-log/054-xs-b-stft-ht-phases-learned.md).
Their checkpoints are gone, so the page does not show them.

## How to rebuild

From the repo root, after `source slurm/<cluster>.env`. Each job clears any stray `LISA_*`
variables, writes its record to `data/` and rebuilds the page. A new run needs no code change.

```bash
# widths and ranks against the per-source MCMC (narrowband cache: narrow runs only), on CPU
sbatch $CANNA_SBATCH_CPU reports/posterior-eval/flow-vs-reference/scripts/widths.sbatch <run>
# the joint-MCMC comparison, once mcmc's joint_mcmc.sbatch chains are cached
sbatch $CANNA_SBATCH_CPU reports/posterior-eval/flow-vs-reference/scripts/compare.sbatch <run>
# the velocity probe on the run's commit; XS/S fit a CPU node, larger runs want a GPU
sbatch $CANNA_SBATCH_CPU reports/posterior-eval/flow-vs-reference/scripts/probe.sbatch <run> [tag] [training]
# a job still training: probe a snapshot of its checkpoint on the job's own GPU
reports/posterior-eval/flow-vs-reference/scripts/live.sh <run> <job id> <tag>
# the page alone (numpy only)
uv run --frozen python reports/posterior-eval/flow-vs-reference/scripts/build_page.py
```

Evaluate the run first (`-m canna.lisa.eval`). For a narrow run, the per-source MCMC cache must
also exist (`-m canna.lisa.mcmc run`).

## Data

- `data/widths/<run>.json` holds per eval source the SNR, sin ι, flow widths, truth ranks and,
  where cached, the per-source MCMC widths. It also records the eval's epoch, weights, ODE steps,
  time_power and draw count.
- `data/joint-<problem id>/compare_window<j>_<run>.{npz,txt}` holds the joint comparison per
  window: the table in `.txt`, and the rows, draws and coverage in `.npz`. The paper's MCMC
  appendix reads the `.npz` (`paper/scripts/fig_mcmc.py`, `fig_mcmc-posterior.py`,
  `numbers_mcmc.py`).
- `data/probe/<tag>.json` holds the velocity probe of one checkpoint. The page shows every epoch
  of a run.

For `dusty-otter-narrow-XS`, the data are: eval `4-average-32` (epoch 4, averaged weights,
1,024 draws, phase reference mid-observation); joint problem `joint-ad8a87a575` (window SNRs 314,
651, 900 and 1607); and the probe at epoch 4 (1,024 samples, CPU).
