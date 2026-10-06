# canna.lisa research log

Scope: the LISA galactic-binary flow (`src/canna/lisa`), all runs on TREX to date
(2026-09-01 to 2026-10-01), and the experiment plan from here. The numbers come from the
slurm logs in `.slurm-logs/` on TREX and from the local evals in `outputs/lisa-XS/`
(gitignored). §8 lists the scripts that produced them. Newest entries go at the top of §4.

## 1. Summary (2026-10-01, updated 2026-10-06)

- **Update, 2026-10-06 (F24).** C2 cooled the 512 net the same way.
  - Its loud floor matches the 768's (×1.03), so the floor is not capacity.
  - The 768 net is 11–15% sharper below SNR 100 and finds 11 more of 840 sources.
  - By the pre-written rule, **S runs at 768** (`S-late-768`, ~34 h, two chained
    slots).
- **Update, 2026-10-06 (F23).** The lr cooldown (C1) is the biggest gain so far.
  - The loud-source f₀ floor goes from 0.087 to 0.037 bins (×0.44, now 25× ideal).
  - The run-specific bias is gone.
  - Faint-source detection beats every earlier model.
  - Calibration stays conservative, and the loss falls 0.022.
  - The constant lr was the main cause of both the floor and the bias.
  - A100 fp32 at autotune level 0 matches the CPU to rounding, which closes F18, F20
    and F22.
  - **Next:** every run gets a cooldown. Then S with the recipe, possibly preceded by
    a 2-h cooldown of the 512 net to choose its width (§6).
- **Update, 2026-10-05 (F22).** On the A100, `--xla_gpu_autotune_level=3` is not enough.
  fp32 posteriors there are 2.4× too narrow and overconfident (in68 0.45). On the laptop,
  the real network and sampler in fp32 match the CPU. fp32 GPU jobs now switch autotuning
  off (level 0), pending one 3-min A100 check. bf16, the production path, is unaffected.
- **Update, 2026-10-05 (F21).** fp32 at eval (CPU) gives the same widths as bf16: ratio
  0.991, 95% interval 0.978–1.000. bf16 at eval is not the floor. The C1 cooldown has
  lowered the loss by 0.012 two-thirds of the way through, more than predicted. Its
  scorecard decides the floor question.
- **Update, 2026-10-05 (F20).** Compiled float32 is wrong on the GPUs because of an XLA GPU
  compilation fault at its default autotuning level. The same program is right at levels 0–3,
  on two jax and two CUDA versions, so it is not our code or the attention.
  `--xla_gpu_autotune_level=3` fixes it on the laptop, and the eval and scorecard jobs now
  set it for fp32 runs. bf16 is unaffected. Whether the A100 is fixed too is a 3-minute
  check.
- **Update, 2026-10-05 (F19).** The 768 × 8 network (E3') is done.
  - The floor is unchanged: 0.087 bins against 0.088, flat from SNR 100 to 1300.
    Capacity is ruled out.
  - It wins back about half of the warp's detection loss (net +11 of 840 sources) and
    lowers the loss by 1.3%.
  - The loud-source f₀ has a bias of about 1/3 of a width. Its shape across the window
    changes from run to run, which points at the constant lr.
  - **Next:** two cheap tests instead of a new run. One is an fp32 scorecard on a CPU node
    (bf16 or not; no GPU). The other is an lr cooldown continuing the 768 model (~3 h).
    Both were submitted on 5 Oct, with their decision rules in §6.
- **Update, 2026-10-03 (F15).** The scorecards settle E2.
  - The warped clock with aux 10% makes posteriors 1.6× narrower at every SNR.
  - It removes the 1M model's low f₀ bias, which is real (8σ).
  - Calibration is conservative, and nothing is over-confident.
  - Detection loses about 2.5% of sources, mostly near SNR 17.
  - The loud-source floor is 0.088 bins, still about 60× ideal, and it is not bf16.
  - **Next:** a free check of the eval's step count (T2), then **one** run: the XS-late
    recipe with the 768 × 8 network (E3).
- **Update, 2026-10-02.** Both 1M-step runs are done.
  - **Uniform clock (E0):** about 22% narrower than 500k and 2 more sources found. The floor
    barely moved, and there is a hint of a low f₀ bias (F12).
  - **Warped clock with aux 10% (E2, XS-late):** trained cleanly, with its loss drop at
    epoch ~60 instead of ~190. Its corner plots look like E0's, but they cannot show the
    floor (F13).
  - **Next:** two short scorecard jobs (T1), which decide whether the warp helped and
    whether either model is calibrated. Then pick the one next training run.
  - **Separately:** do not use the laptop GPU for evals yet (F14).

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
| **13474448** | E2: XS-late training, 1M steps | **Completed** at 03:18 CEST on 2 Oct (10 h 57 min, trexgpu02, 39.6 s per epoch), no NaN. See F13. |
| 13578013 | eval of the 1M model (E0) | Completed in 14 min. Corner plots in `outputs/lisa-XS/corner` on TREX, copied to `outputs/1M/lisa-XS/corner`. |
| 13579022 | eval of XS-late | Completed in 13 min, `time_power 3` read from its config. Corner plots in `outputs/lisa-XS-late/corner`, copied locally to the same path. See F13. |
| 13641222, 13645445 | E3': XS-late-768 | Both died on the home quota at the first and second checkpoint (F16, F17). |
| **13719349** | E3': XS-late-768, 1M steps, `save_opt_state: false` | **Completed** at 04:22 CEST on 5 Oct (16 h 36 min, trexgpu02, 59.6 s per epoch), no NaN. Final flow loss 0.4498. See F19. |
| 13767819, 13767924 | scorecard and eval of XS-late-768 | Completed in 6 and 14 min. Downloaded through JupyterHub (SSH was down) to `outputs/lisa-XS-late-768` (checkpoint 1000, corner plots, `scorecard.npz`). See F19. |
| 13780225 | T3': XS-late scorecard in fp32 on a CPU node (cpu2022, 64 cores, trex085), `--n_random 40` | Running since about 14:20 CEST on 5 Oct, submitted from JupyterHub. TREX code at 30701cf. 88 s per injection on 64 cores, 1 h 15 min in all. Done: same widths as bf16 (F21). |
| 13780234 | C1: XS-late-768-cool, epochs 1000 → 1200 with the lr cooldown | Completed at 17:57 CEST on 5 Oct (3 h 36 min, trexgpu04), final 10-epoch loss 0.4276 (F23). It ran after moving `outputs/lisa-XS-late-768/checkpoints` into `outputs/lisa-XS-late-768-cool/`. It resumed at epoch 1000 with `lr_scale` 0.9975, then 0.9025 at epoch 1020; flow loss still 0.450 there. About 60 s per epoch, ETA about 17:45 CEST. |
| 13790394 | A100 fp32 scorecard of XS-late at autotune level 3 (`--n_random 40`) | Completed in 3 min. It does not match the CPU: 2.4× too narrow, in68 0.45 at SNR ≥ 100 (F22). |
| 13793687, 13793690 | scorecard and eval of XS-late-768-cool (C1) | Completed in 5 and 14 min. Floor 0.087 → 0.037 bins, bias gone, best detection so far (F23). Corner plots copied to `outputs/lisa-XS-late-768-cool/corner`. |
| 13793694 | A100 fp32 scorecard of XS-late at autotune level 0 | Completed in 21 min. Matches the CPU fp32 scorecard to rounding (F23). |
| **13849421** | C2: XS-late-cool, epochs 1000 → 1200 with the lr cooldown | Completed on 6 Oct (2 h 11 min, trexgpu02, 38.8 s per epoch), final 10-epoch loss 0.4352 (F24). |
| 13867994 | scorecard of XS-late-cool | Completed in 4 min. The loud floor matches the cooled 768's; fewer faint sources found (F24). |

## 4. Findings

### F24 (2026-10-06): C2 — once cooled, the 512 net matches the 768's floor but finds fewer sources; S runs at 768

**Runs.**
- C2 training, job 13849421: XS-late (512 × 8) from epoch 1000 to 1200 with C1's cooldown,
  resuming with its optimizer state. 2 h 11 min at 38.8 s per epoch.
- Scorecard 13867994, 4 min, on the same 840 sources as F15, F19 and F23.

**Loss.** The 10-epoch mean went 0.4557 → **0.4352** (−0.020), against −0.022 for the 768
net (0.4276). The gap between the two widths stays about 0.008, as it was before cooling.

| SNR | n | found: 512 at 1M / 512 cooled / 768 cooled | f₀ width, 512 cooled / 768 cooled (bins) | ratio 512c / 768c, median [16–84%] | ratio 512c / 512 at 1M | in68, 512c / 768c | in95 |
|---|---|---|---|---|---|---|---|
| < 15 | 148 | 0.45 / 0.50 / **0.54** | 0.118 / 0.102 | 1.12 [0.96–1.52] | 0.70 | 0.76 / 0.78 | 0.97 / 0.97 |
| 15–40 | 157 | 0.81 / 0.85 / **0.86** | 0.077 / 0.068 | 1.15 [0.96–1.45] | 0.59 | 0.92 / 0.90 | 1.00 / 1.00 |
| 40–100 | 151 | 0.92 / 0.95 / **0.97** | 0.044 / 0.041 | 1.13 [0.91–1.43] | 0.51 | 0.93 / 0.90 | 0.98 / 0.99 |
| ≥ 100 | 384 | 0.99 / 0.995 / 0.995 | 0.039 / 0.037 | **1.03 [0.85–1.32]** | 0.44 | 0.93 / 0.93 | 0.99 / 0.99 |

**The cooldown helps the 512 net exactly as much.** Its loud width falls ×0.443 against its
own 1M checkpoint (C1: ×0.440).
- Its bias goes too: the window-position profile went from 0.045 to **0.009** bins, and the
  loud mean offset is +0.011 ± 0.005.
- Calibration is the same as the 768's.
- So F23's gain is a property of the schedule, not of the wider net.

**The floor does not depend on width.** At SNR ≥ 100 the 512 net is 1.033× the 768's, with a
95% interval of 1.012–1.053.
- By SNR: 0.044 / 0.040 / 0.036 / 0.035 against 0.038 / 0.039 / 0.035 / 0.036 bins.
- So the remaining floor (25× ideal) is not capacity either.

**Width does matter below SNR 100.**
- The 768 net is 11–15% narrower there.
- It finds more sources: 18 found only by the 768 against 7 only by the 512, net 11 of 840
  (sign test p = 0.04). By band: +4.1, +1.3 and +2.0 points.
- That is the same +11 as at constant lr (F19). Width buys detection and intermediate-SNR
  precision, not the loud floor.

**Decision, by the rule written in §6 before the run.** The loud width is within 1.1× (yes,
1.03). The found fractions are within 1 point in every band (no: 4.1, 1.3 and 2.0). So
**S runs at 768**: `configs/S-late-768.yaml`, ~34 h in two chained 24 h slots.

### F23 (2026-10-06): C1 — the lr cooldown halves the floor, removes the bias and finds more sources

**Runs.**
- C1 training, job 13780234: epochs 1000 → 1200, lr decaying linearly to 0, 3 h 36 min.
- Scorecard 13793687 (5 min) and eval 13793690 (14 min).
- The scorecard uses the same 840 sources as F15 and F19. Widths are medians over sources
  that both models found.

| SNR | n | found, 768 at 1M → cooled | f₀ width (bins) | per-source ratio, median [16–84%] | × ideal (cooled) | in68 | in95 | rank |
|---|---|---|---|---|---|---|---|---|
| < 15 | 148 | 0.47 → **0.54** | 0.163 → 0.103 | 0.63 [0.50–0.77] | 2 | 0.80 → 0.78 | 0.97 → 0.97 | 0.49 → 0.53 |
| 15–40 | 157 | 0.83 → **0.86** | 0.119 → 0.067 | 0.53 [0.42–0.67] | 3 | 0.87 → 0.90 | 0.99 → 1.00 | 0.48 → 0.50 |
| 40–100 | 151 | 0.94 → **0.97** | 0.091 → 0.040 | 0.46 [0.34–0.57] | 5 | 0.81 → 0.90 | 0.98 → 0.99 | 0.47 → 0.49 |
| ≥ 100 | 384 | 0.995 → 0.995 | **0.087 → 0.037** | **0.44 [0.35–0.57]** | 25 | 0.82 → 0.93 | 0.98 → 0.99 | 0.41 → 0.49 |

**Loss.** The 10-epoch mean went 0.4499 → **0.4276** (−0.022). At the constant-lr pace of
the last 100k steps (−0.002 per 100k), the same gain would have taken about 1.1M more
steps.

**Floor.** At SNR ≥ 100 the median ratio is **0.440, with a 95% interval of 0.427–0.453**.
- The floor is 0.038 / 0.039 / 0.035 / 0.036 bins at SNR 100–200 / 200–400 / 400–800
  / ≥ 800. It is still flat in SNR, but 2.3× lower: 25× ideal, against 55×.
- Against the rule (≤ 0.07 bins: optimiser noise contributes; ≤ 0.06: it dominates),
  **the constant lr was the main cause of the floor.**

**Bias.**
- The loud-source mean offset went from +0.029 ± 0.005 to **+0.007 ± 0.005** bins, and
  the median |offset| from 0.042 to 0.013 bins.
- The window-position profile, the largest |median offset| over 8 slices, went from 0.081
  to **0.008** bins.
- The rank went from 0.41 to 0.49.
- That passes the rule (≲ 0.03 bins, rank 0.45–0.55). **The run-specific bias of F19 was
  the constant-lr iterate**, as suspected.

**Detection is the best of any model so far.**
- 21 sources are found only by the cooled model, and 1 only by the 1M one.
- Against XS-late it is 33 against 2.
- Against the uniform-clock 1M model (F15), the cooled model now finds more at every SNR:
  0.54 / 0.86 / 0.97 against 0.52 / 0.85 / 0.95.
- So the detection cost of the warp (F15) is gone. Step 3 of §6, `time_power 2`, is no
  longer needed.

**Calibration** is conservative everywhere: in68 0.78–0.93, in95 0.97–1.00. Nothing is
over-confident. At SNR ≥ 100, in68 = 0.93 means the posteriors are still wider than they
need to be.

**Corner plots** (`outputs/lisa-XS-late-768-cool/corner`, local). On the loudest
injection's first-source block, the cooled model has fewer stray contours: the
intermediate-amplitude blob in f₀–A is gone. Sky, ψ and φ₀ look as before. This is a visual
check only; the scorecard measures f₀ alone.

**The A100 fp32 check at autotune level 0** (job 13793694, 21 min; level 3 took 3 min)
reproduces the CPU fp32 scorecard (T3') to rounding:
- widths and offsets agree to a median of 4×10⁻⁷ bins, at most 6×10⁻⁵;
- the found sets and the in68/in95 flags are identical;
- ranks differ by at most one draw in 256.

So on the A100, autotuning at both level 4 and level 3 selects wrong fp32 kernels, and with
autotuning off it is right. **The F22 prediction holds.** fp32 on the GPU is usable again,
at about 7× the eval time. bf16 needs none of this.

### F22 (2026-10-05): level 3 does not fix fp32 on the A100; on the laptop, the real network and sampler are fine

**The A100 test (job 13790394, 3 min).** The XS-late fp32 scorecard on the A100, with
`--xla_gpu_autotune_level=3` and `JAX_DEFAULT_MATMUL_PRECISION=highest`, on the same 200
sources as T3'. **It does not reproduce the CPU:**

| SNR ≥ 100 | CPU fp32 (T3') | A100 fp32, level 3 | A100 fp32, default level (F18) |
|---|---|---|---|
| sources found, all SNR | 169 | 174 | 4 |
| f₀ width | 0.088 bins | 0.036 bins (0.41×) | garbage |
| median distance from the truth | 0.053 bins | 0.037 bins | ~2 bins |
| in68 / in95 | 0.85 / 1.00 | **0.45 / 0.88** | – |

- Level 3 removes the gross failure. But the posteriors are 2.4× narrower while sitting
  about as far from the truth, so they are **overconfident**, not sharper.
- CPU fp32 and A100 bf16 agree with each other and are calibrated (F21). So the A100 fp32
  path is still wrong, now subtly.
- The F20 hypothesis, that level 3 fixes the A100 too, is **refuted**.

**The same checks on the laptop GPU with the real 768 network** (local checkpoint 1000;
`fp32bug/prod_velocity.py`, `prod_sampler.py`):

| | one velocity call, max error / rms relative | full sampler, 256 draws × 32 RK4 steps, max \|draws − CPU\| |
|---|---|---|
| default level, default precision (TF32) | 1.7×10⁻² / 2×10⁻³ | 0.31 |
| default level, `highest` | 1.9×10⁻⁵ / 2×10⁻⁶ | 1.3×10⁻⁴ |
| level 3, `highest` | 1.7×10⁻⁵ / 2×10⁻⁶ | 1.4×10⁻⁴ |
| level 0, `highest` | 1.3×10⁻⁵ / 2×10⁻⁶ | – |

- The real network does not hit F20's gross failure on the laptop, even at the default
  level. That failure belongs to the tiny test network's shapes.
- At full precision, the compiled fp32 sampler matches the CPU on the laptop.
- TF32's 2×10⁻³ per call moves individual draws by up to 0.3 over 128 calls. That alone
  does not mean the distribution is wrong: bf16 is coarser still and gives calibrated
  posteriors.

**So the A100 fp32 fault depends on the GPU, or on the 512 network, and not on the
precision setting.** The A100 picks other kernels, and level 3 still lets it autotune.

**Next, if fp32 on the GPU is still wanted:**
- The eval and scorecard jobs now switch GEMM autotuning off for fp32 (`level=0`). Level 0
  matches the CPU on the laptop for both networks.
- One A100 run of `sbatch slurm/lisa-scorecard.sbatch XS-late --dtype float32 --n_random
  40` (3 min) must match T3' this time.
- If it does not, the A100 fp32 path stays untrusted, and fp32 checks keep running on the
  CPU (88 s per injection).
- Nothing in production depends on it: every run trains and evaluates in bf16.

### F21 (2026-10-05): T3' — fp32 at eval leaves the floor where it is; the cooldown is lowering the loss

**T3' (job 13780225, XS-late, fp32 on a CPU node, 1 h 15 min).** It used the 10 eval
injections plus the first 40 random ones: 200 sources, with the same keys as the bf16 A100
scorecard (F15). Per source, against bf16:

| SNR | n | found, bf16 / fp32 | f₀ width, bf16 → fp32 (bins) | width ratio fp32 / bf16, median [16–84%] |
|---|---|---|---|---|
| < 15 | 33 | 0.39 / 0.39 | 0.154 → 0.141 | 0.98 [0.92–1.00] |
| 15–40 | 41 | 0.78 / 0.78 | 0.132 → 0.131 | 0.99 [0.93–1.03] |
| 40–100 | 31 | 0.94 / 0.94 | 0.098 → 0.093 | 0.99 [0.95–1.03] |
| ≥ 100 | 95 | 1.00 / 1.00 | 0.088 → 0.088 | 0.99 [0.95–1.04] |

- **The loud-source ratio is 0.991, with a 95% bootstrap interval of 0.978–1.000** (93
  sources).
- The decision rule (§6) was ≥ 0.9, so **bf16 at eval is ruled out.** The bf16-noise
  prediction, ≈ 0.56, is excluded. F5's 6% was noise.
- The two runs find exactly the same 169 sources.
- **fp32 does move the medians.** It shifts every loud source by −0.026 ± 0.002 bins,
  about a quarter of a width.
  - The shift is about the same across the window, a little larger at the top. It does not
    depend on the bf16 cell of the input coordinate, so it is not input rounding.
  - Coverage is unchanged (in68 0.86 against 0.85, in95 1.00).
  - The bf16 medians are a little closer to the truth: median |offset| 0.038 against
    0.053 bins. The network was trained in bf16, and that is its best precision for eval.
- **Still open:** bf16 *in training* (F5, Open). It is a costlier test: one run with an
  fp32 head. It comes after C1.
- **Speed:** 88 s per injection on 64 cores, against ~1.6 s on an A100.

**C1 interim (job 13780234, at epoch 1118 of 1200, 16:30 CEST).** The 10-epoch mean flow
loss went from **0.4499** (epochs 991–1000, constant lr) to 0.4490, 0.4471, 0.4450,
0.4422, 0.4400 and **0.4377** (epochs 1101–1110, lr at 0.45).
- That is −0.012 so far. It already beats the predicted −0.004 to −0.010 for the whole
  cooldown.
- It is about 10× faster than the last 100k constant-lr steps (−0.0015).
- So optimiser noise at the constant lr was holding the loss up. Whether it also holds the
  floor up waits for the scorecard. ETA about 17:50 CEST.

### F20 (2026-10-05): compiled float32 is wrong because of an XLA GPU compilation fault, triggered by its default autotuning
This explains F14 and F18. It was found on the laptop GPU (RTX 2000 Ada) with the tiny
float32 test network. The scripts are in `outputs/lisa-XS/eval-tools/fp32bug/`.

**The reproducer is one velocity evaluation.** Eager on the GPU differs from the CPU by
2.9×10⁻⁴. Jit-compiled, it differs by **0.595**, on velocities of at most 0.45.

**The same program gives two answers.** `same_program.py` lowers the jitted velocity once
and hashes its StableHLO: 625 ops, no custom calls. The hash is the same in every run
below. Only the process's `XLA_FLAGS` differ:

| setup | `--xla_gpu_autotune_level` | error vs the CPU |
|---|---|---|
| jax 0.11.0, CUDA 12 (cuBLAS 12.9.2) | default (4) | 0.595 (in two fresh processes) |
| | 3 | 2.9×10⁻⁴ |
| | 0 | 3.0×10⁻⁴ |
| | default, no GPU preallocation | 0.595 |
| jax 0.10.1, CUDA 12 | default | 0.595 |
| | 3 | 3.0×10⁻⁴ |
| jax 0.10.1, CUDA 13 (cuBLAS 13.1.1) | default | 0.595, the same bytes as jax 0.10.1 + CUDA 12 |
| | 3 | 2.9×10⁻⁴ |

- **What this rules out.** The program is fixed, so its meaning cannot depend on a
  performance flag. HLO has no undefined behaviour for these ops, and there are no custom
  calls. So the fault is in how XLA compiles it for the GPU, not in the model code.
- **It is not new and not specific to one version:** jax 0.10.1 and 0.11.0, CUDA 12 and 13.
- **It is not specific to one machine:** the laptop's Ada GPU on driver 580; F18's A100 on
  driver 550 shows the same symptom.
- **It is not memory pressure:** turning preallocation off changes nothing.
- Within one process, repeated calls give the same wrong bytes.

**Bisection.**
- **Not the attention.** On its own it is right under jit. A hand-written attention in its
  place leaves the error. F18's suspect is cleared.
- **Not `scan` or `jax.checkpoint`.** One plain block fails the same way.
- **Program-dependent.** All stages are right when the jitted function returns every
  intermediate, and wrong when it returns only one of them.
- **Inside a block,** the modulation alone is right, while the paths from it into a matmul
  are wrong.
- **Flags:**
  - Autotune levels 0–3 are right, and 4 (the default) and 5 are wrong.
  - Turning off Triton GEMMs, cuBLASLt or command buffers does not fix it.
  - Matmul precision `highest` does not help at level 4.
- **Kernels.** At level 4, one GEMM becomes a cuBLASLt matmul with a BIAS epilogue, the x
  embedding's second Linear. But the same layer compiled on its own is right, and with
  cuBLASLt off the bug stays.
- **The exact faulty step inside XLA is not identified.**

**Related upstream:** [jax-ml/jax#35515](https://github.com/jax-ml/jax/issues/35515).
- float32 on an Ada-generation GPU (RTX 4090), jax 0.9.0, at the default autotune level. The
  level-4 correctness check there rejects every candidate as "WRONG RESULTS".
- Its workarounds include `--xla_gpu_autotune_level=3`.
- It is the same machinery misbehaving, though not necessarily the same bug: there it
  errors, here a wrong plan is used silently.

**bf16 is fine.** On the GPU, compiled and eager bf16 differ by 4.9×10⁻³, about 2 bf16
rounding steps. The A100 bf16 scorecards match the CPU (F15), and the CPU fp32 scorecard
matches them too (F21).

**Workaround on the laptop:** `--xla_gpu_autotune_level=3`, plus
`JAX_DEFAULT_MATMUL_PRECISION=highest` for true fp32 matmuls instead of TF32. It is **not
enough on the A100** (F22), so the jobs now use level 0.
- Together, compiled fp32 matches the CPU to 2×10⁻⁷.
- `slurm/lisa-eval.sbatch` and `lisa-scorecard.sbatch` set both for `--dtype float32`.
- `test_fori_loop_transport_matches_the_unrolled_rk4[3]` still misses its 10⁻⁵ tolerance
  on the GPU, by 1.6×10⁻⁵. That is TF32, not the bug.

**Open:**
- Confirm on the A100: `sbatch slurm/lisa-scorecard.sbatch XS-late --dtype float32
  --n_random 40` must reproduce T3' (F21). About 3 min.
- Report upstream, with `same_program.py` reduced so it no longer needs canna.

### F19 (2026-10-05): E3' — capacity does not move the floor; it wins back half the detection

**Training (job 13719349).** Completed at 04:22 CEST on 5 Oct: 16 h 36 min, 59.6 s per
epoch, no NaN.
- Final flow loss **0.4498**, against XS-late's 0.4557: 0.006 lower, or 1.3%.
- The gap was widest early, 0.023 at epoch 50, because the loss drop came sooner. It is
  0.004–0.008 from epoch 300 on.
- The end slope matches XS-late's: −0.0015 per 100 epochs, against −0.0013.
- The y aux metric grows without bound once its weight reaches 0: 521 here, 1.3×10⁶ in
  XS-late. That head is no longer trained and sampling never uses it, so this is harmless.

**Scorecard (job 13767819, 6 min) and corner plots (13767924, 14 min).** These use the same
840 sources as F15. Widths are medians over the sources both models found:

| SNR | n | found, 1M / late / 768 | f₀ width, late → 768 (bins) | per-source ratio 768 / late, median [16–84%] | in68, late → 768 | in95 | rank |
|---|---|---|---|---|---|---|---|
| < 15 | 148 | 0.52 / 0.45 / 0.47 | 0.149 → 0.158 | 1.00 [0.68–1.28] | 0.78 → 0.80 | 0.99 → 0.97 | 0.50 → 0.49 |
| 15–40 | 157 | 0.85 / 0.81 / 0.83 | 0.128 → 0.117 | 0.92 [0.70–1.18] | 0.95 → 0.87 | 1.00 → 0.99 | 0.46 → 0.48 |
| 40–100 | 151 | 0.95 / 0.92 / 0.94 | 0.088 → 0.089 | 1.00 [0.77–1.17] | 0.89 → 0.81 | 0.99 → 0.98 | 0.47 → 0.47 |
| ≥ 100 | 384 | 0.99 / 0.99 / 0.995 | 0.088 → 0.087 | 0.96 [0.80–1.17] | 0.87 → 0.82 | 0.99 → 0.98 | 0.50 → 0.41 |

**The floor did not move.**
- It is 0.087 / 0.087 / 0.085 / 0.090 bins at SNR 100–200 / 200–400 / 400–800 / ≥ 800.
  XS-late gives 0.089 / 0.091 / 0.087 / 0.087.
- 2.25× the parameters and a 1.3% lower loss leave the floor the same to within 1%.
  Capacity is ruled out, which adds it to the list of F16: the integrator, input rounding,
  A100 numerics and the physics.
- Two candidates remain:
  - **bf16 inside the network (activations and output).** The floor is identical for two
    networks of different width, which is how a fixed numerical resolution behaves. An
    O(1) value in bf16 is resolved to 2⁻⁸ ≈ 0.004, and 0.004 flow units is 0.09 bins (1
    unit = 24 bins): the floor's size. T3 was meant to test this; it is still void (F18).
  - **Optimiser noise at a constant lr** (F9). See the bias below.

**Detection: about half of the warp's loss is back.**
- 16 sources are found only by the 768 net (median SNR 16) and 5 only by XS-late: net +11
  of 840 (sign test p = 0.03).
- Against the 1M uniform-clock model it is 7 against 17: net −10 (p = 0.06).
- On the 10 eval injections, the two nets find and miss the same sources. The SNR 27 and 37
  sources missed by XS-late are still missed.

**Calibration is fine overall, but a bias shows at loud sources.**
- in95 is 0.97–0.99 everywhere.
- in68 moved from conservative (0.87–0.95) towards nominal (0.80–0.87).
- At SNR ≥ 100, though, the rank fell to 0.41. The median sits **+0.021 bins** above the
  truth (mean +0.030 ± 0.005, a 6σ shift), about a third of a width.

**The f₀ bias depends on where the source sits in the window, and its shape changes from
run to run.** Median offset in bins for found sources at SNR ≥ 40, in 8 slices of the f₀
coordinate from −1 to +1 (53–76 sources per slice):

| model | −1 | | | | 0 | | | +1 |
|---|---|---|---|---|---|---|---|---|
| 1M | −0.018 | −0.032 | −0.130 | −0.146 | −0.045 | −0.033 | +0.009 | −0.026 |
| XS-late | +0.018 | −0.021 | −0.027 | −0.018 | −0.015 | +0.005 | +0.043 | +0.045 |
| 768 | −0.015 | −0.017 | +0.039 | +0.081 | +0.063 | +0.003 | +0.001 | −0.009 |

- F15's "no bias in XS-late" was partly a cancellation: about −0.02 in the lower half of the
  window and +0.04 at the top.
- Each run's bias is a smooth error of the learned function, about a quarter of the window
  wide (6 bins), with its own shape and sign. That is where the constant-lr weights happened
  to stop, not something in the physics or the data.
- An lr cooldown, or an EMA of the weights, is the standard cure. It may also be what holds
  the floor up.

**Window-edge pile-up.** XS-late's 4 zero-width sources all sit at |coord| > 0.995. There,
over half the draws share one f₀ value. The 768 net has none of these. They are 4 of 840
and do not move the medians.

**Verdict.**
- Capacity is not the floor.
- The 768 net is a little better than XS-late at 1.5× the cost: 1.3% lower loss and 11
  more sources found, with similar calibration.
- Its loud-source f₀ is biased by about 1/3 of a width, with a run-specific shape.

**The local copy is complete.** `outputs/lisa-XS-late-768/checkpoints/1000` restores on the
CPU: epoch 1000, 170,077,474 parameters, all finite.

### F18 (2026-10-04): float32 compiled LisaFlow is wrong on the A100 too, so T3 is void

**T3 failed as a test.** The XS-late scorecard with `--dtype float32` (job 13719350, done
in 6 min) found almost nothing.
- Found fractions were 0, 1, 2 and 4% across the SNR bands.
- Medians sat 2–5 bins off the truth.
- The same model in bf16 finds 46–99%.

**The cause is the GPU path, not the model.**
- The 500k model evaluated in fp32 on the *CPU* (F5) was fine: all sources found, and the
  loudest one 6% narrower (0.197 bins against 0.210; corrected 2026-10-05 from "wider").
- On the laptop's Ada GPU (F14), jit-compiled float32 LisaFlow was wrong while eager was
  right.

**So F14 is not laptop-specific.** Compiled float32 LisaFlow is wrong on both GPUs.
- The bf16 path is fine. The A100 bf16 scorecard reproduces the CPU table (F15).
- In bf16, `MultiStreamAttention` calls cuDNN flash attention explicitly. In float32 it
  goes to `implementation="xla"`, so XLA's own GPU attention lowering or fusion is the
  first suspect.

**Consequences:**
- Every production result so far (bf16) stands.
- The bf16-output-noise hypothesis is still untested.
- Any fp32 step on a GPU (the roadmap's step 2a) needs this fixed first.

**To test next:**
- a float32 single-block attention, jit vs eager, on the laptop GPU (seconds);
- an fp32 scorecard on the CPU for a few injections.

**E3' is running.** Job 13719349 started 4 Oct at 11:46 CEST on the e7a0864 code, with
`save_opt_state: false`.
- It is past epoch 17, with checkpoints saving: the quota fix works.
- It runs at 59.6 s per epoch, with an ETA of about 04:20 CEST on 5 Oct.

### F17 (2026-10-03): E3' relaunch hit the quota again; checkpoints drop the optimizer state

**What happened.** After the cleanup, job 13645445 saved epoch 1 (`checkpoints/1`), then
failed writing epoch 2's `2.orbax-checkpoint-tmp` (`RESOURCE_EXHAUSTED`).
- The quota holds one 1.7 GB checkpoint, but not the two that orbax keeps during a save.

**Measured speed.**
- Compile takes ~2 min, then **60 s per 1000-step epoch** (epoch 1 at 2:05, epoch 2 at
  3:05).
- So 1M steps is about **17 h**, inside one 24 h job.
- The 57 ms/step estimate held.

**Fix (2026-10-03).** `--save_opt_state/--no-save_opt_state`, set to false in
`XS-late-768.yaml`.
- The optimizer state is 64% of a checkpoint (483 of 771 MB at XS), so a 768-wide
  checkpoint drops from ~1.7 GB to ~0.6 GB.
- `restore_from` notices a checkpoint without optimizer state and keeps a fresh one. It
  prints so in the log.
- `tests/lisa/test_checkpoint.py` covers both ways.
- The cost: a resume restarts muon's momentum and Adam's moments. That only matters at a
  24 h boundary, which this run should not reach.

### F16 (2026-10-03): the integrator is converged, and E3' died on the home-directory quota

**T2: XS-late scorecards at 64 and 128 RK4 steps** (jobs 13641220, 13641221) agree with the
32-step one to within noise, on the 200 random injections.

| steps | found, SNR < 15 / 15–40 / 40–100 / ≥ 100 | f₀ width (bins) | rank |
|---|---|---|---|
| 32 | 0.46 / 0.81 / 0.92 / 0.99 | 0.152 / 0.129 / 0.087 / 0.088 | 0.51 / 0.46 / 0.46 / 0.50 |
| 64 | 0.46 / 0.81 / 0.92 / 0.99 | 0.147 / 0.130 / 0.088 / 0.089 | 0.50 / 0.46 / 0.46 / 0.50 |
| 128 | 0.47 / 0.81 / 0.92 / 0.99 | 0.147 / 0.127 / 0.087 / 0.089 | 0.50 / 0.46 / 0.46 / 0.50 |

- XS-late's detection loss and its 0.088-bin floor belong to the trained network, not to
  the eval's ODE.
- 32 steps stay the default.

**E3' (XS-late-768) failed at its first checkpoint.**
- Job 13641222 trained epoch 1 (129.5 s including compile).
- Orbax's save then failed with `RESOURCE_EXHAUSTED` writing into
  `outputs/lisa-XS-late-768/checkpoints/1.orbax-checkpoint-tmp`, and the job exited with
  code 1.
- The filesystem has 5.8 TB free, so this is the **per-user quota on
  /home/mp/mentasgi**.
- A 768-wide checkpoint is about 1.7 GB, against 0.77 GB at XS, and orbax holds two while
  saving.
- `outputs/` holds eight checkpoint directories, four of them obsolete:
  - `lisa-XS.nan` and `lisa-S.nan` (NaN weights);
  - `aux10` (stopped E1);
  - `lisa-B` (empty);
  - plus the failed temporary checkpoint.
- `500k/lisa-XS` and `lisa-XS` (1M) are backed up on the laptop.
- There is no personal folder under `/work/LISA`, where other LISA users keep theirs.

### F15 (2026-10-03): scorecards — the warp narrows posteriors 1.6× and removes the f₀ bias

**Setup.** Jobs 13626493 (1M, uniform clock) and 13626496 (XS-late), about 4 min each on an
A100. Each covers 840 sources: the 10 eval injections plus 200 random prior injections.
The outputs are in `outputs/scorecards/`.
- "found" means an f₀ width under one bin.
- The ideal width is √3/(π SNR) bins.
- rank is the fraction of draws below the truth; 0.5 means unbiased.
- in68 and in95 are the fractions of truths inside the flow's central intervals.

| SNR | found 1M → late | f₀ width (bins) 1M → late | × ideal 1M → late | median offset (bins) 1M → late | rank 1M → late | in68 / in95, 1M → late |
|---|---|---|---|---|---|---|
| < 15 | 0.52 → 0.45 | 0.223 → 0.152 | 5 → 3 | −0.085 → +0.010 | 0.56 → 0.50 | 0.80 / 0.98 → 0.78 / 0.99 |
| 15–40 | 0.85 → 0.81 | 0.187 → 0.129 | 10 → 6 | −0.042 → +0.021 | 0.58 → 0.46 | 0.87 / 0.99 → 0.95 / 1.00 |
| 40–100 | 0.95 → 0.92 | 0.151 → 0.088 | 18 → 11 | −0.053 → +0.013 | 0.60 → 0.47 | 0.80 / 0.98 → 0.89 / 0.99 |
| ≥ 100 | 0.99 → 0.99 | 0.147 → 0.088 | 94 → 59 | −0.056 → −0.003 | 0.62 → 0.50 | 0.77 / 0.97 → 0.87 / 0.99 |

**Narrower.** On sources found by both, XS-late is narrower by a median factor of:
- 0.61 at SNR ≥ 100, narrower in 97% of them;
- 0.62 at SNR 40–100;
- 0.70 at SNR 15–40.

**F12's low f₀ bias is real in the 1M model and gone in XS-late.**
- With 840 sources, the 1M ranks of 0.56–0.62 are about 8σ away from 0.5 for the 369 loud
  random sources.
- XS-late's ranks are 0.46–0.51, with offsets of about 0.
- Both models are on the *conservative* side: in95 is 0.97–1.00, and in68 sits above
  0.68. No over-confidence.

**Detection got slightly worse.** 27 sources are found only by the 1M model (median SNR 17)
and 6 only by XS-late, a net −21 of 840. This is the expected cost of the warp: about 21% of
training samples land in the first half of the path, against 50% with a uniform clock.
- Part of it may come from the eval instead. With uniform steps in t, the first half of
  the path (s < 0.5) gets only ~6 of the 32 RK4 steps.

**The floor moved, but it is still a floor.** At SNR ≥ 100 the width is 0.088 bins, still
about flat in SNR (Spearman −0.16) and about 60× ideal.
- **It is not bf16.** The width does not track the bf16 cell of the source's f₀ coordinate
  (Spearman −0.11, the wrong sign). Sources with a 0.023-bin cell have 0.089 bins, those
  with a 0.094-bin cell 0.086.

**The A100 reproduces the CPU table** (F12) on the 31 localised eval sources: median width
ratio 0.97, one outlier (the collapsed q0.40 source).
- So F14's laptop-GPU problem does not reach TREX.

**Against E2's decision rule:**
- The floor fell 1.7×, not ≥ 3×, to 0.088 bins rather than ≲ 0.06.
- Calibration improved.
- B fell slightly.
- **The warp is a keeper.** It narrows everything and removes the bias. But it is not
  enough on its own.

### F14 (2026-10-02): on the laptop GPU, compiled code disagrees with eager code
- **The failing test.** `test_fori_loop_transport_matches_the_unrolled_rk4` fails on the
  laptop's RTX 2000 Ada (after the driver fix) and passes on the CPU.
- **Bisection, on the tiny float32 test network.**
  - Eager evaluation on the GPU matches the CPU to 3×10⁻⁴.
  - *Any* jit-compiled version, the eval's `fori_loop` or a jitted copy of the unrolled
    RK4, is off by 0.63 in flow units, about 2/3 of the whole transport.
  - Matmul precision `highest` does not change it.
  - The network takes the xla attention path here (float32), not cuDNN.
- **TREX is not affected, at least not visibly.** In the A100 corner plots, loud-source f₀
  and sky marginals are sharp spikes on the truth. An error that size would smear f₀ by
  ~15 bins.
- **Consequence:** do not trust laptop-GPU evals or training until this is understood.
  - Running `scorecard.py` on the 1M model on TREX also checks the A100. Its ten eval rows
    use the same keys and draws as F12's CPU table, so they must reproduce it.
- **Open:** a minimal reproducer on the laptop, then a JAX/XLA version check.

### F13 (2026-10-02): XS-late training and eval, first look (no numbers yet)

**Training.**
- 1M steps took 10 h 57 min, with no NaN.
- The loss has a fast drop at **epoch ~60** (−3×10⁻³ per epoch, aux weight 0.35). The
  500k baseline's came at ~190, and E1 (aux 10%, uniform clock) had none by epoch 57.
  - So the drop is not set by the aux schedule.
  - It fits it being the network learning the late-path (fine-scale) structure, which the
    warp gives more weight to.
- The 500k → 1M doubling lowered the loss by 0.014, the same pace as E0.
- The loss values are not comparable to E0's, because the warp re-weights the objective.

**Eval (corner plots only).**
- Side by side with the 1M model on q0.30, q0.60 and q1.00 the two look alike:
  - posteriors sit on the truth;
  - nothing visibly collapses;
  - the same sources are found;
  - XS-late's 2D sky contours are somewhat more scattered.
- **The corner plots cannot answer the question E2 asks.** Each f₀ axis spans the whole
  48-bin window, so a 0.15-bin and a 0.03-bin posterior both draw as a spike. The verdict
  on the floor (A), detection (B) and calibration (C) waits for `scorecard.py` (T1).

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

## 6. Roadmap (2026-10-03, updated 2026-10-06)

### Update, 2026-10-06 (after C2, F24)

**C2 has decided the width: S runs at 768.** The steps below shift up by one.
1. **S1: `S-late-768`.** It is the next GPU job, about 34 h.
   - Its predictions, written now:
     - the loss falls at the cooldown as on XS;
     - the loud-source f₀ floor sits near XS's 0.04 bins (25–30× ideal);
     - SNR 15–40 detection is at least 80%;
     - in95 is ≥ 0.95 in every band.
   - **If the floor is ≥ 2× XS's,** the wider window costs precision. The next lever is
     then the S-specific conditioning: 4× the tokens per window.
2. **MCMC validation** (jexplore) on 2–3 XS injections, using `XS-late-768-cool`. It can run
   on the laptop while S trains, with the user's go: it is a CPU job of a few hours.
3. **Lower priority:** bf16 in training; a longer cooldown; B.

### Update, 2026-10-06 (after C1, F23)

| | XS 1M, uniform | XS-late | XS-late-768 | **XS-late-768-cool** |
|---|---|---|---|---|
| loud-source f₀ width (SNR ≥ 100) | 0.147 bins | 0.088 | 0.087 | **0.037** (25× ideal) |
| found, SNR < 15 / 15–40 / 40–100 | 52 / 85 / 95% | 45 / 81 / 92% | 47 / 83 / 94% | **54 / 86 / 97%** |
| loud f₀ bias | −0.056, rank 0.62 | ±0.04 by position | +0.021, rank 0.41 | **+0.002, rank 0.49** |
| in68 / in95 at SNR ≥ 100 | 0.77 / 0.97 | 0.87 / 0.99 | 0.82 / 0.98 | 0.93 / 0.99 |
| GPU time | 10.8 h | 10.9 h | 16.6 h | 16.6 + 3.6 h |

**Settled.**
- The lr schedule was the main lever. A warmup-stable-decay schedule (the last ~20% decaying
  to 0) goes into every run from now on.
- Neither bf16 at eval nor network width moved the floor at a constant lr (F19, F21).
- Faint-source detection no longer needs its own run.

**Still open:**
- The floor is still flat in SNR, at 25× ideal for loud sources. The posteriors are
  conservative (in68 0.93), so they are wider than the data require.
- Whether width matters once the lr is annealed. F19 tested capacity at a constant lr only.

**Next, in order:**
1. **C2, optional: cool down XS-late (512) the same way.** That is 200k steps, about
   2.2 h. It decides the network width for S:
   - S at 512 takes ~80 s per epoch, so 1M steps is ~22 h, one 24 h slot;
   - S at 768 takes ~123 s per epoch, ~34 h over two slots.
   - **Rule:** if the cooled 512 net's loud width is ≤ 1.1× the cooled 768's (≤ 0.041
     bins), and its found fractions are within 1 point per band, S uses 512. Otherwise S
     uses 768.
   - Needs: a config like `XS-late-768-cool.yaml`, and the XS-late checkpoint backed up
     locally first, because the run replaces it on TREX.
2. **S with the recipe.** `time_power 3`, aux 10%, 1M steps of which the last 200k cool
   down, and the width from C2. This is the run that carries the method to the wider band.
3. **Validation against MCMC** (jexplore) on 2–3 injections. Now that the flow is sharper,
   this measures how far its conservative widths sit from the true posterior. It also
   extends the calibration check beyond f₀ (sky, amplitude).
4. **Lower priority:**
   - bf16 in *training* (an fp32 head);
   - a longer cooldown or total, if the floor matters after step 3;
   - B, which needs a /work folder.

### Update, 2026-10-05 (after E3', F19)

| | XS 1M, uniform clock | XS-late | XS-late-768 |
|---|---|---|---|
| loud-source f₀ width (SNR ≥ 100) | 0.147 bins | 0.088 bins | 0.087 bins |
| found, SNR < 15 / 15–40 / 40–100 | 52 / 85 / 95% | 45 / 81 / 92% | 47 / 83 / 94% |
| loud f₀ bias | −0.056 bins, rank 0.62 | ≈ 0 overall, ±0.04 by position | +0.021 bins, rank 0.41 |
| in95 | 0.97–0.99 | 0.99–1.00 | 0.97–0.99 |
| GPU cost (1M steps) | 10.8 h | 10.9 h | 16.6 h |

**Ruled out as causes of the floor:** the integrator, input rounding, A100 numerics, the
physics, and now capacity (F19).

**Two candidates are left. Both tests were submitted on 2026-10-05, with no new 1M-step
run.**
1. **T3': the XS-late scorecard in fp32 on a CPU node** (`slurm/lisa-scorecard-cpu.sbatch`,
   10 eval injections plus 40 random, no GPU time). XS-late stands in for the 768 net: they
   share the floor, the cheaper net runs about 2× faster on a CPU, and its checkpoint is
   not touched by C1.
2. **C1: an lr cooldown on the 768 model** (`configs/XS-late-768-cool.yaml`, ~3.3 h of
   A100). It resumes at 1M and trains 200k more steps, with the lr decaying linearly from
   1e-4 to 0.
   - The decay is a per-epoch scale on the update (`--cooldown_steps`). With no weight
     decay that is exactly a scaled lr, and it survives a resume with a fresh optimizer
     state.
   - It leaves the best model so far, whatever the outcome.
   - It overwrites the 1M checkpoint on TREX. The local copy in
     `outputs/lisa-XS-late-768/checkpoints/1000` is the only one left.

**Predictions and decision rules (written 2026-10-05, before the results):**
- **T3'.** The statistic is the per-source width ratio, fp32 on the CPU over bf16 on the
  A100, for loud found sources (SNR ≥ 100, about 90 of them).
  - **Reference.** The 1M model's bf16 widths agree between the CPU and the A100 to a
    median ratio of 0.97 (F15).
  - **If bf16 adds the noise F5 hinted at:** F5's 0.210 → 0.197 bins implies ≈ 0.073 bins
    added in quadrature. Removing that from 0.088 leaves **≈ 0.05 bins, a ratio of
    ≈ 0.56**.
  - **Ratio ≤ 0.75:** bf16 sets most of the floor. Next is fp32 in the velocity head and
    last block. That needs one training run, and evals need the GPU float32 bug fixed (F18)
    or must stay on the CPU.
  - **Ratio ≥ 0.9:** bf16 at eval is ruled out. A network *trained* in bf16 could still
    be limited (F5, Open). That is a costlier test and comes second.
  - My prior: about 1 in 3 that the ratio falls below 0.75. F5 is a single source, at 16
    steps.
- **C1.**
  - **Loss** falls 0.004–0.010 over the cooldown, to 0.440–0.446. That is the usual
    warmup-stable-decay gain.
  - **If optimiser noise holds the floor up:** the loud width falls at least 20%, to
    ≤ 0.07 bins. The position-dependent bias shrinks from a largest slice offset of 0.081
    bins to ≲ 0.03, and the loud rank returns to 0.45–0.55.
  - **If the floor stays at 0.085–0.09 and the bias stays:** the constant lr is ruled out.
  - Detection: at most +1%.
- **Then:**
  - **One test positive:** its fix goes into the S run, which is one run with the whole
    recipe (step 4 below). That is the 768 net, `time_power 3`, aux 10%, a cooldown over
    the last 20%, plus the fp32 head if T3' says so.
  - **Both negative:** the floor is either bf16 *in training* or the objective itself
    (F6). The S run then goes ahead with the recipe as it is. The floor question moves to a
    single fp32-head training test, and validation against MCMC (step 5) decides how much
    the floor matters.
  - **Detection (step 3) waits.** The 768 net recovered half the gap, and `time_power 2`
    would trade back some of the floor.

### Where things stand (2026-10-03)

| | XS 1M, uniform clock | XS-late, warped clock + aux 10% |
|---|---|---|
| loud-source f₀ width (SNR ≥ 100) | 0.147 bins, 94× ideal | 0.088 bins, 59× ideal |
| found, SNR 15–40 / < 15 | 85% / 52% | 81% / 45% |
| f₀ bias | low, 8σ | none |
| calibration (in95) | 0.97–0.99 | 0.99–1.00, conservative |

**Ruled out as causes of the floor:**
- the ODE integration (32 = 64 = 128 steps, F16);
- bf16 rounding of the *input* coordinate (F5, F15);
- f₀–ḟ physics (F12);
- A100 numerics (F15).

**Not yet tested:**
- bf16 noise in the network's *output* and internal activations (T3 below);
- capacity (E3', running);
- the learning rate and its schedule (F9).

**Why bf16 output noise is now a suspect.**
- The velocity the network outputs is O(1) in flow units. bf16 keeps 8 significant bits,
  so its rounding is ~0.002–0.004 units, which is 0.05–0.1 bins.
- That noise would be flat in SNR and blind to where the source sits in the window,
  exactly like the floor.
- The fp32 eval of the 500k model in F5 moved its width by only 6%. But with a 0.2-bin
  floor, an added ~0.07-bin noise *should* move it by only ~6%, so that test could not
  see it. At XS-late's 0.088 bins it would dominate.

### Steps, each with its decision

1. **T3: XS-late scorecard in fp32 (eval only, ~10 min on an A100).**
   Command: `sbatch slurm/lisa-scorecard.sbatch XS-late --dtype float32`, which writes
   `scorecard_float32.npz`.
   - **Loud widths drop clearly** (≲ 0.06 bins): bf16 noise is the floor. Go to step 2a.
   - **Otherwise:** bf16 is ruled out for good. Go to step 2b.
2. **Precision or capacity.**
   - **2a: fix the precision.** Evaluate in fp32 (free). Then make the velocity head, and
     if needed the last block, compute in fp32 during training. That is a small code
     change, with a cost near zero because most of the network stays in bf16. Then check
     whether E3' (bf16) also gains from an fp32 eval.
   - **2b: capacity.** E3' (768 × 8, job 13645445) decides it. Its scorecard is compared
     with XS-late's: floor and detection.
3. **Detection of faint sources.**
   - If E3' does not recover the −2.5% at SNR ≈ 17, the next training run uses
     `time_power: 2` instead of 3. That puts 29% of samples in the first half of the path,
     against 21%.
   - The same run carries an lr cooldown over its last 20% (F9). Per the compute rule, it
     is one combined run.
4. **S** (the 0.1–4.2 mHz band, 128 tokens) with the settled recipe.
   - One run of ~1M steps. At 768 wide that is probably ~30 h, so two 24 h job slots
     (resume works).
   - B (the full band) needs a /work allocation and a memory plan first.
5. **Validation against a real sampler.** jexplore MCMC on 2–3 injections, chosen as one
   crowded window with a missed SNR ~25–35 source and one loud window. Then extend the
   scorecard's calibration to sky and amplitude.

**Housekeeping:**
- ask the `/work/LISA` admin (owner `palacih`) for a personal folder: the home quota is
  tight (F16);
- report or minimise F14 (compiled JAX code wrong on the laptop's Ada GPU);
- keep running the lisa tests with `JAX_PLATFORMS=cpu` meanwhile.

## 6b. Experiment plan, as run (revised 2026-10-01: compute-limited, as few runs as possible)

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
| E0 | XS 500k → 1M, then its eval | Q2: does more training move A or B? | eval only (~15 min) | done: width table F12 (A 92× → 76×, B 8 → 10 of 13), corner plots 13578013 |
| E1 | XS, `warmup_frac 0.1` | Q3, and what triggers the jump (F10) | — | stopped at epoch ~60; only F10 survives |
| **E2** | **one combined run**: XS-late, 1M steps (details below) | Does the recipe fix the floor (A) and the misses (B)? | ~10.8 h + eval | done (F13, F15): 1.6× narrower, bias removed, detection slightly worse; the floor is still ~60× ideal |
| E3 | S with the E2 recipe | the wider band | ~15 h | only if E2 succeeds |
| **T1** | `scorecard.py` on the 1M model and on XS-late: per-source width and ideal, found, offset, 68%/95% coverage, rank; 10 eval + 200 random injections | the E2 verdict, the calibration question from F12, and an A100 cross-check of F12 | 2 short eval jobs (~4 min each) | done (13626493, 13626496), F15 |
| T2 | XS-late scorecard again at 64 and 128 RK4 steps (`--ode_steps`, eval only; writes `scorecard_ode<N>.npz`) | Do the detection loss and the 0.088-bin floor come from the integrator? The warp leaves the first half of the path only ~6 of 32 steps | 2 × ~5 min | done (13641220, 13641221): no, the integrator is converged (F16) |
| **E3'** | `configs/XS-late-768.yaml`: the XS-late recipe (`warmup_frac 0.1`, `time_power 3`, 1M steps) with a **768 × 8** network, 12 heads of 64 (170M parameters) | Does capacity lower the floor and win back detection? | ~16 h (est. 57 ms/step; resumable if the 24 h limit hits) | done (13719349, 16 h 36 min; F19): same floor (0.087 bins), net +11 sources found, loss −1.3%, a run-specific loud-source f₀ bias of ~1/3 width |
| **T3'** | the XS-late scorecard in fp32 on a TREX **CPU** node (`slurm/lisa-scorecard-cpu.sbatch`, `--n_random 40`), since GPU fp32 is broken (F18); writes `scorecard_float32_cpu.npz` | Does bf16 set the 0.087–0.088-bin floor? XS-late and the 768 net share the floor, so the cheaper net answers it | none (CPU, 1 h 15 min) | done (13780225, F21): width ratio fp32 / bf16 0.991 [0.978–1.000], bf16 at eval ruled out |
| **C1** | `configs/XS-late-768-cool.yaml`: the 768 model continued from 1M to 1.2M steps, with the lr decaying linearly to 0 over the 200k (`--cooldown_steps`). Its checkpoint is moved, not copied, into `outputs/lisa-XS-late-768-cool`; `require_checkpoint` stops it from starting from scratch | Does optimiser noise hold up the floor and cause the run-specific bias (F9, F19)? | ~3.3 h | done (13780234, F23): floor 0.087 → 0.037 bins, bias gone, detection up; loss −0.022 |
| **C2** | `configs/XS-late-cool.yaml`: XS-late (512 × 8) continued from 1M to 1.2M steps with the same cooldown as C1; its checkpoint (with optimizer state) is moved into `outputs/lisa-XS-late-cool` | Does width matter once the lr is annealed? Rule: loud width ≤ 1.1× C1's (≤ 0.041 bins) and found fractions within 1 point per band → S at 512 | ~2.2 h | done (13849421, 13867994; F24): loud width ×1.03 the 768's, but found −4.1 / −1.3 / −2.0 points, so S at 768 |
| **S1** | `configs/S-late.yaml` (512) or `S-late-768.yaml`: S with the full recipe, 1M steps, the last 200k cooling down | Does the recipe carry to the 0.1–4.2 mHz band? | ~34 h in two chained slots (768, per F24) | ready: `S-late-768` |

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

On TREX, from `~/canna`, after pulling the latest bundle (TREX cannot reach GitHub):

```bash
# when a run's .out ends with [done]:
sbatch slurm/lisa-eval.sbatch XS                              # E0 (1M model, outputs/lisa-XS)
sbatch slurm/lisa.sbatch XS-late                             # E2 -> outputs/lisa-XS-late, ~10.8 h
sbatch slurm/lisa-eval.sbatch XS-late                        # its eval, after [done]
sbatch slurm/lisa-scorecard.sbatch XS                        # T1 on the 1M model -> outputs/lisa-XS/scorecard.npz
sbatch slurm/lisa-scorecard.sbatch XS-late                   # T1 on XS-late
sbatch slurm/lisa-scorecard.sbatch XS-late --ode_steps 64    # T2 -> outputs/lisa-XS-late/scorecard_ode64.npz
sbatch slurm/lisa-scorecard.sbatch XS-late --ode_steps 128   # T2 -> scorecard_ode128.npz
sbatch slurm/lisa.sbatch XS-late-768                         # E3' -> outputs/lisa-XS-late-768, ~16 h
# after E3' prints [done]:
sbatch slurm/lisa-scorecard.sbatch XS-late-768               # compare with XS-late's scorecard (F15)
sbatch slurm/lisa-eval.sbatch XS-late-768                    # corner plots
# T3' and C1 (F19), independent of each other:
sbatch slurm/lisa-scorecard-cpu.sbatch XS-late --dtype float32 --n_random 40   # -> outputs/lisa-XS-late/scorecard_float32_cpu.npz
mkdir -p outputs/lisa-XS-late-768-cool
mv outputs/lisa-XS-late-768/checkpoints outputs/lisa-XS-late-768-cool/          # a move: the quota has no room for a copy
sbatch slurm/lisa.sbatch XS-late-768-cool                                        # C1, 1000 -> 1200 epochs, ~3.3 h
# after C1 prints [done]:
sbatch slurm/lisa-scorecard.sbatch XS-late-768-cool
sbatch slurm/lisa-eval.sbatch XS-late-768-cool
# C2 (F23, §6): cool down the 512 net the same way
mkdir -p outputs/lisa-XS-late-cool
mv outputs/lisa-XS-late/checkpoints outputs/lisa-XS-late-cool/                 # a move: C2 replaces it
sbatch slurm/lisa.sbatch XS-late-cool                                            # 1000 -> 1200 epochs, ~2.2 h
# after C2 prints [done]:
sbatch slurm/lisa-scorecard.sbatch XS-late-cool                                  # compare with XS-late-768-cool
# S1 at 768 (F24), ~34 h: a chain of two jobs, the second resuming from the first's
# checkpoint when the 24 h limit hits
JOB=$(sbatch --parsable slurm/lisa.sbatch S-late-768)
sbatch --dependency=afterany:$JOB slurm/lisa.sbatch S-late-768
# after [done]:
sbatch slurm/lisa-scorecard.sbatch S-late-768
sbatch slurm/lisa-eval.sbatch S-late-768
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
- **E3' (F19).** `outputs/lisa-XS-late-768/` is the TREX output folder. It was downloaded on
  2026-10-05 as a tar through JupyterHub and holds `checkpoints/1000` (no optimizer state),
  `corner/`, `scorecard.npz` and `losses.pdf`.
  - `outputs/scorecards/XS-late-768.npz` is a copy of the scorecard.
  - `outputs/scorecards/compare_768.py` makes the F19 tables, including the bias by window
    position.
  - The slurm logs 13719349, 13767819 and 13767924 are in `outputs/lisa-XS/logs/`.
- **Code on TREX without SSH (2026-10-05).** Uploading a file through JupyterHub did not
  reach the home folder. What worked was pasting a base64 git bundle into the JupyterHub
  terminal: `outputs/paste-into-trex.txt` (bundle `e7a0864..30701cf`, md5
  118bb29a963c3d7b34460aebbbf2faa3), then `git pull --ff-only ~/canna-30701cf.bundle fml`.
  TREX is now at 30701cf.
- **T3' and F20 (2026-10-05).**
  - `outputs/scorecards/XS-late_float32_cpu.npz` is the CPU fp32 scorecard, fetched over
    sftp. `t3_cpu.py` there makes the F21 tables.
  - `outputs/lisa-XS/eval-tools/fp32bug/` holds the GPU bisection: `step1`–`step6`,
    `minimal*`, `bf16_check`, and `same_program.py` for the fixed-program test.
  - The jax 0.10.1 venvs (CUDA 12 and 13) were built offline from the uv cache in the
    session scratchpad, and are not kept.
- **F22.** `outputs/scorecards/XS-late_float32_gpu_at3.npz` is the A100 level-3 run, and
  `a100_fp32.py` there makes the comparison. `fp32bug/prod_velocity.py` and
  `prod_sampler.py` hold the laptop checks with the real network.
- **C1 (F23).** `outputs/scorecards/XS-late-768-cool.npz` (also copied into
  `outputs/lisa-XS-late-768-cool/`) and `compare_cool.py` hold the cooled scorecard.
  `XS-late_float32_gpu_at0.npz` is the A100 level-0 check. The corner plots and
  `losses.pdf` are in `outputs/lisa-XS-late-768-cool/`. The cooled checkpoint (epoch 1200)
  is on TREX only so far.
- **Where the large files live (2026-10-06).** The user's rule is that large files are
  kept only on the PC. Checkpoints and corner plots are downloaded from TREX into the same
  `outputs/<run>/` layout locally, verified, and then removed from TREX by the user.
  - `outputs/lisa-XS-late/checkpoints/1000`: XS-late at 1M, with the optimizer state.
  - `outputs/lisa-XS-late-768-cool/checkpoints/1200`: the cooled 768 model, the best so
    far.
  - `outputs/lisa-XS-late-768/checkpoints/1000`: the 768 model at 1M.
  - `outputs/1M/lisa-XS/checkpoints/1000`: XS with the uniform clock at 1M.
  - On TREX only small files stay: scorecards, `losses.pdf` and logs.
- **C2 (F24).** `outputs/lisa-XS-late-cool/` holds `checkpoints/1200` (params only, verified
  on the CPU), `scorecard.npz` and `losses.pdf`, also copied as
  `outputs/scorecards/XS-late-cool.npz`. `compare_c2.py` makes the F24 table and applies the
  rule.
