# canna.lisa research log

Scope: the LISA galactic-binary flow (`src/canna/lisa`), all runs on TREX to date
(2026-09-01 to 2026-10-01), and the experiment plan from here. The numbers come from the
slurm logs in `.slurm-logs/` on TREX and from the local evals in `outputs/lisa-XS/`
(gitignored). §8 lists the scripts that produced them. Newest entries go at the top of §4.

## 1. Summary (2026-10-01)

- **Where XS stands (500k steps).** The posteriors are centred on the truth and capture the
  ψ + kπ/2, φ₀ + π and A–ι degeneracies. They have two problems:
  - **Loud sources hit an f₀ floor of 0.13–0.25 bins.** It is flat from SNR 60 to 1400,
    so it sits at 18–410× the Fisher width (median 92×).
  - **Detection is incomplete.** At SNR 15–40, 5 of 13 sources are not localised to
    within a bin. Below SNR 15, none of the 6 are.
- **The floor is not numerical.** The ODE is converged by 32 steps, fp32 changes it by 6%,
  and it does not track the bf16 spacing of the coordinate.
- **The floor is invisible to the training loss.** Under uniform-t flow matching, a
  0.18-bin floor on every loud source costs about 2×10⁻⁴ of logged flow loss. That is below
  the epoch-to-epoch scatter and about 1% of what a doubling of training steps buys. The
  optimiser has essentially no reason to remove it.
- **Q3, aux schedule at 10% of the budget: E1 partly answered it before it was stopped (F10).**
  - In the baseline the flow loss crept down at 2×10⁻⁴ per epoch, then fell 0.056 in
    35 epochs. That drop came as the aux weight crossed ~0.2, at epoch 175 of 500.
  - E1 crossed 0.2 at epoch 36 and did *not* drop. It is only slightly ahead of the
    baseline (0.007 lower at epoch 57).
  - So the jump is not simply caused by the aux weight. What triggers it is open until
    E1 finishes.
- **Q2, more training time: modest and diminishing; wait for the 1M eval.**
  - The loss improves by about 0.017 per doubling of steps. The forecast is 0.405 ± 0.002
    at 1M steps and about 0.395 at 2M.
  - That improvement acts on the coarse errors, not on the floor.
  - A learning-rate cooldown should capture much of a doubling's gain at no extra cost.
- **Q1, bigger network: include it in the one combined run (E2).**
  - It is cheap here: 768 × 8 costs about 1.5× per step.
  - After F10 a capacity limit on the flow can no longer be ruled out.
  - XS, S and B all use the same 75.6M-parameter network. S is a bigger problem, not a
    bigger model.
- **The lever that targets the floor is the time distribution:**
  - emphasise t → 1 by sampling 1 − t log-uniformly;
  - condition the network on log(1 − t);
  - integrate on a time grid that is fine near t = 1.
- **Plan (compute-limited):** read out E0, the run already paid for. Then launch **one**
  combined run (E2) carrying every change: `warmup_frac 0.1`, the 768 × 8 network,
  late-time emphasis and an lr cooldown. Then S, with that recipe, only if E2 works.

## 2. Setup (`fml` @ dff3ef2)

| | XS as trained |
|---|---|
| problem | 4 sources per window, T_obs = 2 yr, `response_points: 16` → a 2 × 64 × 3 (A/E/T) WDM image. Sources are drawn in the central 48 of the 64 bins. f₀ ∈ [0.1, 1.3] mHz, which is 53% of the log band. The SNR prior is 7–1000. |
| flow coordinates | 11 per source: f₀ and A are log-whitened to [−1, 1] over the window (1 unit = 24 bins); log M_c is standardised; the sky lives on S², (ψ, ι) on S², φ₀ on S¹. A `Set` geometry over the 4 sources assigns each draw to its nearest-source permutation. |
| network | `LisaFlow`: MMDiT with hidden 512, 8 blocks, 8 heads, expand 2. It sees 32 y-tokens and 4 x-tokens. **75.6M parameters, the same in XS, S and B.** It has three heads: the velocity, an x₁ prediction (`u_pred`), and a clean-data reconstruction (y). |
| objective | Σᵢ wᵢ Lᵢ / Var(targetᵢ), with w = (1, a, a). a runs a cosine from 1 to 0 over `warmup_frac` × `total_steps` (0.5). t ~ U(0, 1), straight geodesic path, no σ_min. |
| optimiser | global-norm clip at 1, then `optax.contrib.muon` (0.2.8, default width scaling) at lr = 1e-4. Adam, at the same lr, handles the non-matrix parameters. **Constant lr, no schedule**, no weight decay. bf16 compute, fp32 parameters, batch 256, 1 epoch = 1000 steps. |
| the rungs XS / S / B | They differ only in `response_points` (rp), the number of frequency bins of each source's response that jaxgb computes. The window is 4·rp bins, so the image is 2 × 4rp WDM pixels and the network sees 2rp y-tokens. A source's Doppler sideband plus its chirp drift must fit, and that caps f₀. **XS:** rp = 16, a 64-bin window, 32 tokens, f₀ ≤ 1.3 mHz (53% of the log band). **S:** rp = 64, 256 bins, 128 tokens, f₀ ≤ 4.2 mHz (78%). **B:** rp auto-sized to 1024, 4096 bins, 2048 tokens, the full 0.1–12 mHz. Everything else is the same: network, 4 sources, priors, optimiser, 500k steps. |
| cost (A100-80G) | XS takes 38.9 s per epoch, about 23 ms fixed plus 15 ms of network per step. S takes 80 s per epoch. |
| eval | 10 injections at the 0.1…1.0 SNR quantiles of 1024 prior draws (window SNR 121–1848), 1024 flow draws each, RK4 with 32 steps. The Fisher overlay is the Hessian averaged over 32 noise realisations, plus the prior precision. |

## 3. Run history

| job | what | outcome |
|---|---|---|
| 10876424, 10876803 | B | OOM at compile (32 GiB). Then a cuDNN CUDA-graph failure, fixed with `XLA_FLAGS=--xla_gpu_enable_command_buffer=`. |
| 10999774, 10999969 | XS | `configs/XS.yaml` was not yet on the cluster. |
| 11000056 / 11000586 | XS / S, 500k steps each | Ran all 500 epochs, but went NaN at epoch 55 (XS) and 62 (S), abruptly, from a flat loss. Cause: the x-loss `log_map` orientation. Fixed in 2e71862, together with abort-on-non-finite. Kept as `outputs/lisa-XS.nan`, `lisa-S.nan`. |
| 11000386, 11109701 | eval | One crash (numpy `.at`) and one hang. Both fixed. |
| 12486854 | XS, 30-min test | Reached epoch 43, finite, tracking the old run's loss. |
| **13264195** | XS, epochs 43 → 500 | Completed, final flow loss 0.420, no NaN. Backed up to `outputs/500k/lisa-XS` on TREX and `outputs/lisa-XS/checkpoints/500` locally. |
| **13377710** | eval of the 500k model, 32 steps | OK. Plots in `outputs/500k/lisa-XS/corner`, local copy in `outputs/lisa-XS/corner-gpu`. |
| **13378657** | E0: XS continuation 500k → 1M (aux stays 0) | **Completed** at 04:58 CEST on 1 Oct (5 h 25 min). Final flow loss **0.4061** at 1M steps, inside the 0.403–0.408 forecast (F8). Not yet evaluated. |
| 13395354 | E1: XS, `--warmup_frac 0.1`, 500k steps, `outputs/aux10` | **Stopped by hand** around epoch 60 (about 02:40 CEST on 1 Oct) to save compute. Its last checkpoint stays in `outputs/aux10`. See F10. |
| 13399477, 13473447 | E2 (XS-late) training, then its eval | **Both failed at start** (24 s and a few s): `configs/XS-late.yaml` and the time-warp code had not been copied to the cluster, so no XS-late training happened. Fix: push, then pull on TREX, then resubmit. |

## 4. Findings

### F12 (2026-10-01): E0 posterior readout — more training helps a little, and accuracy may slip

Method: the same 10 injections, keys, 256 draws and 32 RK4 steps as F4, run on the laptop
CPU (`widths_ckpt.py`, output `samples1M/`, comparison in `compare_500k_1M.npy`).

| | 500k | 1M |
|---|---|---|
| localised (< 0.8 bin), SNR 15–40 | 8 / 13 | 10 / 13 |
| localised, SNR ≥ 40 | 21 / 21 | 21 / 21 |
| median f₀ width, SNR ≥ 60 | 0.177 bins (92× Fisher) | 0.156 bins (76× Fisher) |
| width ratio 1M / 500k, sources localised in both | — | median 0.78 (IQR 0.72–0.87); narrower in 27 of 29 |
| sky widths / Fisher, SNR ≥ 20 (λ, β) | 10.9, 10.2 | 8.1, 8.5 |
| truth within ±1 own width of the median | 90% | 72% |
| median f₀ offset, localised SNR ≥ 20 | −0.045 bins (19 of 29 low) | −0.083 bins (23 of 31 low) |

- **F6's prediction for E0 was partly wrong.**
  - Training on did shrink every width, by about 22% for 2× the compute, and found 2 more
    moderate-SNR sources.
  - The loud-source floor still moved only from 92× to 76× Fisher. It is not a practical
    route to Fisher-level widths.
- **Accuracy may be slipping as the widths shrink.** f₀ medians sit low of the truth, and
  the shift grew from 500k to 1M.
  - **It is not physical.** The chirp drift over T_obs is about 10⁻⁴ bins (at most
    5×10⁻³), so an f₀–ḟ trade-off cannot shift f₀ by 0.08 bins. The offset does not
    correlate with the flow's chirp-mass error (r = −0.06).
  - **It is not a pull towards the window centre** (15 of 31 point that way, Spearman 0.00),
    so it is not samples stopping short of their target. It is a uniform shift to lower f₀.
  - **Its significance is modest.** Sources in one injection share noise and conditioning.
    At injection level, 8 of 10 have a negative median offset (p = 0.11). It needs ~100+
    injections to settle.
- **One collapse.** q0.40, source 3 (SNR 32.5) went from 0.060 bins to 0.009 bins, which is
  *narrower than its Fisher width* (0.02), and sits 9 widths off the truth. Over-confidence
  is the failure mode to watch as the widths shrink, and it matters most for XS-late.

### F11 (2026-10-01): E0 training readout — the loss forecast held
- **The 1M run completed** (13378657, 5 h 25 min). The final flow loss is 0.4061; the mean
  of epochs 991–1000 is 0.4064.
- **The forecast was accurate.** The power law fitted on epochs 300–681 predicted 0.4070.
  Its rms residual over epochs 682–1000 is 7×10⁻⁴, about the epoch scatter.
- **The 500k → 1M doubling bought −0.0137.**
- **Refit on epochs 300–1000:** L∞ = 0.378, α = 0.56. That gives L(2M) = 0.392–0.397, i.e.
  −0.009 to −0.014 for one more doubling.
- **Open:** whether the posteriors improved. The eval and the per-source widths of the 1M
  model are still to come.

### F10 (2026-10-01, 02:34 CEST): E1 early readout — no jump when the aux weight crosses 0.2

E1 is job 13395354: `--warmup_frac 0.1`, otherwise identical to the baseline, same seed. In
E1 the aux weight crossed 0.2 at epoch 36 and reached 0 at epoch 50.

| epoch | E1 aux | E1 flow | baseline aux | baseline flow |
|---|---|---|---|---|
| 12 | 0.885 | 0.601 | 0.995 | 0.593 |
| 24 | 0.563 | 0.564 | 0.979 | 0.563 |
| 36 | 0.206 | 0.548 | 0.952 | 0.550 |
| 45 | 0.035 | 0.541 | 0.925 | 0.545 |
| 57 | 0 | 0.534 | 0.881 | 0.541 |

- **Prediction E1(i) failed.** There is no fast drop at the crossover.
  - Since then E1's flow loss has fallen about 5×10⁻⁴ per epoch. That is faster than the
    baseline's 3.6×10⁻⁴ at the same epochs, but far from the 2.5×10⁻³ of the baseline's drop.
  - E1 leads by 0.007 at epoch 57.
  - The epoch-12 gap is noise. Two runs of the same config (11000056 and 12486854)
    differed by up to 0.04 at epoch 10.
- **F7's mechanism is wrong as stated.** The aux weight alone does not trigger the jump.
  Two explanations remain:
  - the jump needs something that takes training time (about 175k steps in the baseline,
    where it happened to coincide with the aux decay);
  - or it needs the long aux phase to have happened first.
- **E1 was stopped at epoch ~60, so this stays open.** Had it run on, it would have told them apart:
  - A drop in E1 before epoch ~175 means extra flow weight speeds the transition up.
  - A drop near epoch 175 means training time sets it.
  - No drop, and a 500k loss above 0.420, means the long aux phase was doing real work.
    That is the two-stage idea in `notes.tex`.
- **The capacity question is open again (Q1).** F7's observation still holds: the aux heads
  did not degrade during the drop. But that only shows that no trade-off with the aux
  heads was needed. It does not rule out a capacity limit on the flow itself.

### F9 (2026-10-01): the optimiser takes very small steps and never anneals
- Muon at lr 1e-4 in optax's default width scaling gives an update RMS of about
  (0.03–0.045)·lr ≈ 4×10⁻⁶ per element per step.
  - That is about 5× smaller than AdamW at the same nominal lr, whose updates have
    RMS ≈ 0.2·lr.
  - It is about 200× smaller than the lr ≈ 0.02 that muon's reference setting uses.
- The lr is also constant, so no cooldown ever banks the final loss.
- Neither point is tested yet. This is E3; E5 adds a cooldown.

### F8 (2026-10-01): the pure-flow phase is a slow power law
- Loss decrease per doubling of steps: 250k → 500k gave −0.022; 300k → 600k gave −0.018.
- Fits of L∞ + A·s^−α from epochs 260–400 onward give:
  - L∞ = 0.37–0.39, α = 0.44–0.86;
  - L(1M) = 0.406–0.408 and L(2M) = 0.396–0.401.
- A log-linear fit gives 0.403–0.404 at 1M.
- L∞ is the asymptote of *this recipe*, not the Bayes floor of the problem.
- The epoch-median scatter is 5×10⁻⁴, so differences of about 2×10⁻³ between runs at
  matched steps are real.

![XS training curves](figures/xs-training-curves.png)

### F7 (2026-10-01): the long aux phase delays flow learning, and it is not a capacity limit

Stitched XS lineage (12486854, then 13264195, then 13378657), 9-epoch means:

| epoch | aux weight | flow | x head | y head |
|---|---|---|---|---|
| 43 | 0.93 | 0.545 | 0.184 | 0.042 |
| 170 | 0.24 | 0.514 | 0.172 | 0.036 |
| 175 | 0.21 | 0.512 | 0.172 | 0.036 |
| 190 | 0.14 | 0.485 | 0.168 | 0.036 |
| 210 | 0.065 | 0.456 | 0.163 | 0.037 |
| 250 | 0 | 0.442 | 0.162 | 0.040 (1.0 by epoch 260) |
| 500 | 0 | 0.420 | 0.216 | 12 |

- **Before the drop.** From epoch 43 to 170 the flow loss falls at 2–2.7×10⁻⁴ per epoch.
- **The drop.** Once the aux weight crosses ~0.2 at epoch 176, the rate jumps 12×, to
  2.5×10⁻³ per epoch. The loss falls 0.056 in 35 epochs, then settles onto the F8 power
  law.
- **No capacity trade-off.** Through the drop the x head improves (0.172 → 0.163) and the
  y head holds (0.036 → 0.037). Freeing capacity from the aux tasks would show up as the
  aux losses degrading while the flow improves, and they do not. The y head only degrades
  once its weight is *exactly* 0, and even a weight of 0.005 kept it at 0.038. Keeping the
  aux tasks is cheap; what hurt was their *weight*.
- **Likely mechanism (a hypothesis).** Muon orthogonalises the summed gradient, so every
  step has a fixed size and the loss weights only set its *direction*. While
  a·‖∇L_aux‖ ≳ ‖∇L_flow‖, most of each step went to the aux heads.
- **Consequence.** `warmup_frac: 0.5` spends the first 35% of the budget in that regime.
  With 0.1, the crossover moves to epoch ~35 of 500.
- This answers the "the jump means the network is small" hypothesis in `notes.tex`: the
  data point the other way.
- **Revised by F10.** E1 did not jump when its aux weight crossed 0.2, so the mechanism
  above is wrong as stated, and capacity is not ruled out.

### F6 (2026-10-01): the loud-source floor is nearly invisible to the flow-matching loss

**The model.** One coordinate of one source, in 1-D Gaussian form:
- x₀ ~ N(0, s₀²), with s₀ = 0.58, the std of U[−1, 1];
- x₁ ~ N(0, σ²), the true posterior;
- a model whose field is exactly that of a N(0, σ_θ²) posterior.

**The result.** The excess loss is E|v_θ − v*|² ≈ s₀·σ_θ per coordinate, whatever the true
σ. The flow loss rewards a posterior for being narrow in absolute terms, not for being close
to the true width.

**What the current floor costs.**
- σ_θ = 0.18 bins (0.0075 flow units) on the f₀ of roughly half of all sources (those
  above SNR ~60) costs about **1.7×10⁻⁴** of logged flow loss. The loss is averaged over
  4 sources × 11 coordinates.
- Compare that with the epoch scatter (5×10⁻⁴) and with what a doubling of steps buys
  (1.7×10⁻²). No amount of training at the current weighting will push on it.
- **Bigger networks and longer runs can't help either.** They lower the part of the loss
  that is visible to the objective, which is the coarse structure: detection, sky and
  amplitude.

**Why it is invisible: time sampling.**
- The structure that sets a σ-wide posterior lives at 1 − t ≲ σ/s₀. For the current floor
  that is 1 − t ≲ 0.014, which is 1.4% of the uniform-t samples. For Fisher-width loud
  sources it is 1 − t ≲ 3×10⁻⁵.
- Sampling 1 − t log-uniformly on [10⁻⁶, 1] weights the same floor **29×** more. A 50/50
  mixture with uniform t gives **15×**. (Script: `loss_sensitivity.py`.)

**The time embedding also has to change.** `SinusoidalEmbed` angular frequencies run only
from 1 to 2π per unit t. The network therefore cannot tell 1 − t = 10⁻³ from 10⁻⁴, so
emphasising late times needs log(1 − t) conditioning as well.

### F5 (2026-10-01): the floor is not numerical
- **ODE integration.** The width of the loudest source is 0.76 / 0.42 / 0.27 / 0.22 / 0.21
  bins at 4 / 8 / 16 / 32 / 64 RK4 steps. It is converged by 32.
- **Eval precision.** fp32 compute at eval gives 0.197 bins against 0.210 in bf16, at 16
  steps.
- **Input quantisation (bf16).** The bf16 cell of a source's f₀ coordinate ranges from 0.012
  bins near the window centre to 0.094 bins near the edges. The width does not track it:
  Spearman ρ = −0.11 (p = 0.64) over the 21 sources above SNR 50. Width also does not track
  |coordinate| or SNR.
- **Open.** A network *trained* in bf16 could still have learned only features coarser
  than the noise its activations carry. After E2 this test should be repeated: if the
  floor then tracks the bf16 cell size, precision has become the limit.

### F4 (2026-09-30): what the 500k XS model gets right and wrong

These are the 40 sources of the 10 eval injections, at 32 RK4 steps with 256 draws. Width is
the MAD of f₀ in bins of 1/T_obs; the full table is in
`outputs/lisa-XS/corner-gpu/per_source_widths_32steps.txt`.

| SNR band | n | not localised (> 0.8 bin) | median f₀ width | median width / Fisher |
|---|---|---|---|---|
| < 15 | 6 | 6 | 4.0 bins | 72× |
| 15–40 | 13 | 5 (3 prior-wide at ~9 bins, 2 at ~1 bin) | 0.36 | 14× |
| 40–60 | 1 | 0 | 0.15 | 15× |
| ≥ 60 | 20 | 0 | 0.18 (0.13–0.25 for 18 of the 20; 0.34 and 0.61 at SNR 62–64) | 92× (18–410×) |

- Sky widths are 1.5–56× Fisher.
- Offsets of the flow median from the truth: for 15 of the 20 loud sources they are within 0.8 of the
  flow's own width, and 13 of the 20 are low, so there is a slight low bias of the loud-source f₀.
- Degeneracies are captured.
- The 3 prior-wide sources at SNR 17–37 all sit in crowded windows.

![f0 width vs SNR](figures/xs-500k-f0-width-vs-snr.png)

### F3 (2026-09-30): eval fixes
- **ODE steps.** `ODE_STEPS` went from 4 to 32, with the RK4 loop moved into `lax.fori_loop`.
  At 4 steps the plots were about 1000× broader than Fisher, and integration error alone
  accounted for 3.5× of that.
- **Fisher overlay.** It is now inverted in the Jacobi-scaled frame, since the raw diagonal
  spans 45 decades, and it is eigen-sampled with each eigenvalue floored at the prior's
  precision along that direction. It is a Gaussian forecast centred on the truth by
  construction, not an MCMC. For unmeasured parameters such as M_c below 1 mHz it looks
  like a measurement when it is only the prior width.
- **Restore.** `restore_from` now works for a GPU checkpoint restored on the CPU.

### F2 (2026-09-17 to 29): the NaN fix holds
- The x loss is now `log_map(target, pred)`. The old argument order scaled with the
  prediction's radius.
- 13264195 passed the old blow-up epochs (55 and 62) and finished all 500 epochs finite.
- Early losses reproduce the old run to within ±0.003 by epoch 43.

## 5. The three options

### Q1: a bigger network
- **No evidence of a capacity limit.**
  - The loss "jump" came with the aux heads *also* improving (F7).
  - The loss is still falling steadily at constant lr, which means the run is limited by
    optimisation, not saturated.
- **It cannot touch the main defect.** The loud-source floor is invisible to the loss
  (F6), so extra capacity would not be spent on it.
- **Where it could help:** separating sources in crowded windows (the prior-wide misses at
  SNR 17–37) and the sky widths.
- **Cost is low here**, because about 60% of the step is fixed overhead. Cost scales as
  blocks × hidden²:

| network | params | FLOPs vs XS | est. ms/step | 500k steps |
|---|---|---|---|---|
| 512 × 8 (current) | 75.6M | 1× | 38 | 5.3 h |
| 512 × 16 | 143M | ~2× | ~53 | ~7.4 h |
| 768 × 8 | 170M | 2.25× | ~57 | ~7.9 h |
| 1024 × 8 | 302M | 4× | ~83 | ~11.5 h |

- **S and B are not bigger networks.** They are wider frequency windows: 128 and 2048
  tokens, covering 78% and 100% of the band. Running S now would reproduce the same two
  problems at twice the cost.
- **Verdict (revised after F10):** put it in the combined run E2 rather than a separate
  ablation. Its effect on the coarse metrics will not be separable from the cooldown's,
  and that is accepted to save a run.

### Q2: more training time
- **The run is already in flight** (E0, 500k → 1M). The forecast is 0.405 ± 0.002 at 1M;
  a further doubling would bring about −0.01.
- **It will not remove the floor** (F6). What it can improve is the localisation of
  moderate sources and the sky widths.
- **Verdict:** judge it from the E0 eval, on the same 10 injections. Do not commit longer
  runs until then. The cheaper way to bank "more time" is an lr cooldown at the end of a
  run, which goes into E2.

### Q3: aux for ~10% of the budget
- **Status:** E1 was stopped at epoch ~60. The evidence it left (F10) is weaker than F7
  suggested: no jump at the crossover, only a small lead (0.007 at epoch 57).
- **Choice for E2:** `warmup_frac 0.1`. It is no worse so far and leaves the flow 90% of
  the budget.
  - The risk is that the long aux phase is what enables the jump.
  - E1b (no aux) is dropped to save compute.

## 6. Experiment plan (revised 2026-10-01: compute-limited, as few runs as possible)

**Principle.** Run no single-variable ablations beyond the two already paid for. One
combined run carries every change we believe in, and the runs in flight supply most of the
attribution.

**Scorecard**, reported for each run on the same 10 eval injections (seed 0):
- **A.** Median f₀ width / Fisher of the SNR ≥ 60 sources (500k baseline: 92×).
- **B.** Localised fraction at SNR 15–40 (500k baseline: 8 of 13).
- **C.** Median |offset| / width, as a bias and calibration check. A narrower posterior that
  misses the truth is worse, not better.
- **D.** Flow loss at matched steps (500k baseline: 0.420). This is only informative for
  runs with the same objective; E2 is judged on A–C.

| id | what | answers | new GPU cost | status |
|---|---|---|---|---|
| E0 | XS 500k → 1M, then its eval | Q2: does more training move A or B? | eval only (~15 min) | done; width table F12 (A 92× → 76×, B 8 → 10 of 13); cluster corner plots not yet run |
| E1 | XS, `warmup_frac 0.1` | Q3, and what triggers the jump (F10) | — | stopped at epoch ~60; only F10 survives |
| **E2** | **one combined run**: XS-late, 1M steps (details below) | Does the recipe fix the floor (A) and the misses (B)? | ~10.8 h + eval | **running**: job 13474448 on trexgpu02 since 16:22 CEST on 1 Oct, ~39.6 s per epoch, ETA ~03:30 CEST on 2 Oct |
| E3 | S with the E2 recipe | the wider band | ~15 h | only if E2 succeeds |

**E2 contents (prepared 2026-10-01 as `configs/XS-late.yaml`):**
- **Network:** the XS network, unchanged (512 × 8, 75.6M parameters). The user chose not
  to enlarge it yet.
- **Aux schedule:** `warmup_frac 0.1`, so aux runs for the first 100k steps.
- **Emphasis near t = 1 by a warped clock, `time_power: 3`.**
  - t stays uniform, and so does the network's t input. The point is placed at
    s = 1 − (1 − t)³ along the geodesic.
  - The target is the path velocity d/ds. The eval ODE multiplies it by
    ds/dt = 3(1 − t)².
  - Equivalent to an s-density ∝ (1 − s)^(−2/3), much milder than log-uniform. It weights
    the 0.18-bin floor 15× more and a 0.02-bin blur 59× more.
  - The share of samples at s < 0.5 drops from 50% to 21%. Over 1M steps that is about
    210k early-time samples, against 250k in the 500k baseline.
  - Uniform RK4 steps in t crowd towards s = 1: the last of 32 ends at 1 − s = 3×10⁻⁵.
  - No change to the time embedding: conditioning on t already stretches 1 − s by the
    third power.
- **Length:** 1M steps at constant lr 1e-4 (no cooldown). It compares directly with E0
  at 1M.
- **Estimated time:** about 10.8 h of training (38.9 s per 1000 steps; the warp costs
  nothing), plus about 40 min for the eval.

**Attribution without extra runs:**
- **For the floor (A):**
  - E0 shows whether more flow training moves it.
  - The aux schedule is not separated now that E1 is stopped. It is not expected to
    matter for A (F6).
  - F6 predicts that the network size will not.
  - If E2 moves A, the credit goes to the late-time emphasis.
- **For the coarse metrics (B, sky):** E2 is compared with E0 at the same 1M steps. The
  aux schedule and the clock warp are not separated. That is accepted.
- **If E2 fails**, the late-time change is the first suspect, since it is the only new
  code. The bisection costs one rerun of E2 without it.

**Dropped to save compute:**
- E1b (no aux);
- the muon lr screen (F9 stays untested; lr is the one lever left alone);
- a separate network-size run;
- a separate long run.

**Predictions and decision rules**, written before the results:
- **E0.** Flow loss at 1M is 0.403–0.408. A stays at about 92× (the floor at 0.13–0.25
  bins). Any gain shows up in B and the sky widths.
  - If A and B are unchanged from 500k, longer runs are not the bottleneck.
- **E1.** (i) *failed* (F10). (ii) and (iii) were never tested, because the run was
  stopped.
- **E2.**
  - The loud-source floor falls at least 3×, to ≲ 0.06 bins, and starts to scale with SNR.
  - B is at least E1's.
  - C stays below 1.
  - If the floor falls but then tracks the bf16 cell size, repeat F5 and move the
    x-embedding and the velocity head to fp32. That needs no new run if checked at eval.

## 7. How to run

On TREX, from `~/canna`, which is at dff3ef2:

```bash
# when a run's .out ends with [done]:
sbatch slurm/lisa-eval.sbatch XS                              # E0 (1M model, outputs/lisa-XS)
sbatch slurm/lisa.sbatch XS-late                             # E2 -> outputs/lisa-XS-late, ~10.8 h
sbatch slurm/lisa-eval.sbatch XS-late                        # its eval, after [done]
```

## 8. Provenance

Everything below is gitignored, under `outputs/lisa-XS/`.

- **Logs.** `logs/canna-lisa-*.out|err` are the TREX slurm logs, fetched on 2026-10-01 at
  01:32 CEST.
- **`eval-tools/` scripts:**
  - `parse_logs.py` parses the logs into `xs_lineage.npy`, `xs_old.npy` and `s_old.npy`.
  - `fit_curve.py` does the transition rates and the power-law fits (F7, F8).
  - `bf16_floor.py` does the bf16-cell test (F5) and writes `bf16_floor_rows.npy`.
  - `loss_sensitivity.py` is the Gaussian loss model (F6).
  - `figs.py` makes the figures.
  - `floor_test*.py` covers the ODE-steps and fp32 tests.
  - `widths32.py` produces the per-source table (F4).
- **Samples.** `eval-tools/samples/*.npz` hold the flow and Fisher samples, the truth and
  the Fisher σ; `eval-tools/samples32/*.npy` hold the 32-step flow samples.
