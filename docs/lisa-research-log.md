# canna.lisa research log

Scope: the LISA galactic-binary flow (`src/canna/lisa`), all runs on TREX to date
(2026-09-01 to 2026-10-01), and the experiment plan from here. The numbers come from the
slurm logs in `.slurm-logs/` on TREX and from the local evals in `outputs/lisa-XS/`
(gitignored). §8 lists the scripts that produced them. Newest entries go at the top of §4.
§9 places the work in the literature (survey of 2026-10-09).

## 1. Summary (2026-10-01, updated 2026-10-14)

- **Update, 2026-10-14 (F41): the TREX reads.**
  - **XS:** the average x confirms F40's regression: 682 of 840 found, 1.5–1.6× XS-fd's
    width.
  - **B-sfplus's x matches B1's final model** (loud 3.24 bins against 3.17, better
    calibrated) in 10 h against 85 h. The ~3-bin floor is still flat in SNR.
  - **On B,** the bf16 cast of the source embedding resolves f₀ only to 2.7 bins in the worst
    case, right at the floor. A float32 scorecard (14505803) tests it.

- **Update, 2026-10-13 evening (F40): the schedule-free fixes work, the run regressed.**
  - **Mechanism fixed:** z stays bounded (0.013 against 0.49), and the average x beats the
    training weights (−0.009 in held-out loss).
  - **XS-sfplus's training weights are worse than XS-fd's** on the same 840 sources: f₀
    1.5–1.7× wider at every SNR, 673 found against 732, loud floor 0.015 bins. Calibration
    stays good.
  - **Optimizer regime:** the 1/L1 weighting, coupled to the decay, drove the weights 3.6×
    below their initial size, and the gradient clip binds throughout.
  - **Confounded:** that regime, the cut to 4 tokens and the SNR ordering cannot be told
    apart from one run. B-sfplus's scorecards are on TREX (the SSH session closed).

- **Update, 2026-10-13 night (F39): the fixes are in.** Our own schedule-free wrapper has:
  - decay 0.05 on z;
  - Muon momentum 0.8 and AdamW b1 0.9 inside;
  - averaging weights t·lr² and a warm-started average;
  - β annealed from 0.9 to 0.965.

  Tested, smoke-run on the laptop, not yet trained at scale.

- **Update, 2026-10-13 (F38): B-fd's last read, at epoch 45 of 200 (scorecards of x and y,
  and the eval). The run was cancelled at epoch 48 (F35), so this is the read it ended on.**
  - **Loud f₀ 9.5 bins**, 3.3× B1's *final* 3.2. Nothing found yet (< 1 bin), and faint
    sources are not localised.
  - **But the width now falls with SNR** (log-log slope −0.28, B1's −0.05). The run is
    unbiased and conservative, and x and y are equal in f₀.
  - **Loss:** 0.018 behind B1 at the same epoch and closing 3.5× faster, so it would have crossed
    B1 near epoch 50–55.
  - **Verdict:** undecided. It was on a better trajectory than B1, but the rule (< 1 bin)
    needed another ~10× it never got to try. It also ran the schedule-free setup that F34–F36
    found broken.

- **Update, 2026-10-13 night (F37): the literature.**
  - No flow-matching or diffusion paper uses schedule-free.
  - SF-NorMuon (arXiv:2605.23061) reports our failure: without weight decay on z, z
    diverges and contaminates x. Decay at y, which is what optax does, is also unstable.
  - ScheduleFree+ (arXiv:2605.19095) weights later iterates more (r = 1) for long runs.
  - **To do before the next schedule-free run:** decay on z (our own wrapper), and r = 1 or
    an annealed β.

- **Update, 2026-10-13 night (F36): why schedule-free failed.**
  - Schedule-free steps z with Muon's fixed-size updates. With no weight decay, z ran away to
    twice the weights' size, and its loss is 2.7×10²¹.
  - Only a thin neighbourhood of the training point y = 0.9x + 0.1z is a good network. The
    average x sits on the steep side of it (0.467 against 0.4575 at y; 85 at 30% of the way
    to z).
  - Next run: weight decay ~0.05, and evaluate an EMA or a cooldown of y, or test whether
    schedule-free works once z is held.

- **Update, 2026-10-13 night (F35): the comparison was stopped at 9k steps.**
  - Decay 0.05 cuts the backbone's weight growth by ~40% at no cost in loss.
  - Schedule-free's z drifts from its average at the same rate with or without decay.
  - Early on, the average x beats the training weights y, as intended.
  - Muon momentum inside schedule-free and schedule-free versus plain Muon are not yet
    tested.
  - B-fd was cancelled at epoch 48.

- **Update, 2026-10-13 (F34): no real weight decay; the cost of Muon; ordered sources.**
  - **Weight decay.** 1e-6 times the lr is ~0 per step, so XS-fd's backbone weights grew
    19× (R1 3–4×), and z ended further from its average than the weights' own size.
    Schedule-free averaged a wide cloud. The training weights y are no sharper: loud floor
    0.012 bins, unbiased.
  - **Muon's cost** is Newton–Schulz in float32 (21 ms per XS step on an A100); in bf16 it
    is 12 ms.
  - **Fewer tokens** make B 4.9× (patch 8) to 9× (patch 16) faster per step.
  - **Ordered sources.** Sources are now ordered loudest first, the `Set` geometry is gone,
    and the reconstruction head is back. Fixed labels make transport paths 38% longer.
  - **Next run** waits on the user's choice of weight decay, sort key and patch.

- **Update, 2026-10-12 night (F33): XS-fd matches R1 except at the loud end, where the
  missing cooldown is the main suspect.**
  - **Below SNR 100** XS-fd ties R1: f₀ widths ×0.93–1.07, the same detection (−8 of 840)
    and coverage.
  - **Loud floor:** **0.0103 bins (7.5× ideal)**, 1.54× R1's 0.0067 and 3.8× better than
    the same WDM net without position features (0.039).
  - **Against the MCMC:** loud f₀ is 8.7–13.5× wide; A, sky and orientation are unchanged
    at 3–9×. So the bf16 rounding of the data (F32) was not their limit.
  - **Schedule-free's average lags.** The evaluated weights x trail the training weights y
    by **0.0118 ± 0.0004** of flow loss on identical samples. x averages the whole run,
    including the weights from before the late loss drop (epochs 950–1100). y is closer
    to R1 (+0.0116), but its loud f₀ is no sharper. Neither weight set is the equivalent
    of a cooled model, and R1's cooldown took it from 0.010 bins at mid-run to 0.0067.
  - **B-fd** is at epoch 40 of 200, still 0.024 behind B1's loss but closing.
  - **Pending:** the scorecard of y (14401076, queued) and B-fd. Next step proposed in §6.

- **Update, 2026-10-12 (F32): a new recipe, launched on XS and B together.** The user
  stopped B2 overnight and set a new direction.
  - **Data:** the whitened rfft window (real and imaginary parts of A, E and T) replaces
    arcsinh(WDM).
  - **Embeddings:** piecewise linear encodings (PLE) of every input scalar, computed in
    fp32.
  - **Optimiser:** schedule-free Muon. Muon at lr 1e-3, RMS-matched; AdamW at 3e-4 with
    b2 0.95 for the encodings and the head; weight decay 1e-6.
  - **Out:** the aux heads, the cooldown and the position features.
  - **Two findings behind it:**
    - every earlier run trained 89% of its parameters, the whole MMDiT backbone, with
      Adam, not Muon;
    - the bf16 cast of arcsinh(WDM) rounded the loudest pixels by 0.5–3.6 noise σ.
  - **Runs:** XS-fd (14326113, 52 s per epoch, so ~17 h) and B-fd (14326117 → 14326121, ~4 days) started at
    04:22 CEST, each with its scorecard and eval chained. Their rule is in §6.

- **Update, 2026-10-11 afternoon (F31): R1's final model is the best XS model.**
  - **f₀ floor:** the loud floor is **0.0067 bins, 4× ideal** (26× without the features):
    5.3× narrower per source, with near-nominal coverage (in68 0.71–0.78), no bias, and 7
    more sources found, at the 512's cost.
  - **Against the MCMC:** loud f₀ goes from 38–84× to 5–12×. But A, sky, inclination and
    phases stay 3–9× too wide, and the same six crowded faint sources are missed.
  - **Coverage:** 125/128 pairs inside the 95%. The new misses are a loud source biased by
    a missed neighbour 0.16 bins away.
  - **Next:** B2 (running, ~87 h). Then one XS run (R2) for the remaining precision limit
    of the other coordinates (§6).

- **Update, 2026-10-11 (F28–F30): the position features work; B2 is running.**
  - **R1 at mid-run (F30).** A scorecard at epoch 474 of 1200, before its cooldown, puts the
    loud f₀ floor at **0.010 bins** (95% CI 0.009–0.011): 7× ideal, against 0.039 (26×)
    without the features. That passes the rule (≤ 0.012).
    - The floor falls with SNR again, and coverage is close to nominal.
    - It finds 31 fewer of 840 sources so far, and its loss lags its baseline by 0.05. At
      init the features are 7× the source embedding, so the next run zero-initialises them.
  - **B2 (`B-pos`, 14275499 → 14275503) started at 04:15 CEST** on the strength of that read:
    87 h, plus scorecard and eval.
  - **MCMC on windows q0.30 and q0.80 (F28).** The flow misses 6 of 16 sources, which the
    MCMC localises. Five sit within 2.6 bins of a 6–90× louder source. Coverage holds:
    127/128 pairs inside the 95%.
  - **Crowding (F29).** The nearest source is a median 5.2 bins away. Detection at SNR 15–40
    drops from 98–100% beyond 5 bins to 63% within 1 bin, and to 63% within 5 bins of a
    3×-louder neighbour.
  - **TREX:** commands now run from here, through a shell fed on stdin (§7).
  - **Paper:** a related-work appendix, a draft abstract, and the results above.

- **Update, 2026-10-10 evening (F27): flow against MCMC on two XS windows** (jexplore,
  32-D joint, two converged seeds each).
  - The flow is never biased: the MCMC median is inside its 95% in 64 of 64
    source–parameter pairs.
  - Loud sources are too wide: f₀ 38–84× the MCMC width, everything else 4–8×.
  - Found faint sources are near-optimal (1–2×).
  - In each window it misses one faint source the MCMC localises (SNR 27.1: 231× in f₀),
    which costs a brighter neighbour ~20% of its draws.
  - Fisher matches the MCMC at SNR ≥ 567, so the floor's "× ideal" holds.
- **R1 is ready** (f55ab93, `configs/XS-pos.yaml`, PositionedLisaFlow), handed to TREX on
  10 Oct.

- **Update, 2026-10-10 (F26): B1 trains, but does not localise f₀.**
  - **Training:** 200k steps in four 24 h links, no NaN. Flow loss 0.473; the cooldown
    gave −0.025.
  - **Widths:** f₀ is 2.8–4.5 bins wide at every SNR, so only 1 of 240 sources is under a
    bin. The posteriors over-cover (in68 0.84–0.90). The sky is resolved, but ψ and φ₀ are
    not.
  - **Cause:** in the flow's f₀ coordinate, B's floor is XS's: 2.0×10⁻³ against
    1.5–1.6×10⁻³ units of x_f. One unit is 24 bins on XS and 1536 on B. So the
    loud-source floor is a fixed precision of the coordinate, and B's window makes it
    3 bins.
  - **Ruled out:** the bf16 input cell, the bf16 y-token positions (only 641 distinct of
    2048 on B, a defect in its own right) and eval-time rounding.
  - **Next (§6):** multi-scale fp32 Fourier features for the source coordinate and the
    token positions, tested on XS first (one ~13 h run), then B.

- **Update, 2026-10-09 (§9): related work.** A survey of neural and simulation-based
  inference for LISA, with 39 entries added to or updated in `paper/references.bib`.
  - **Closest work:** Delmond, Korsakova et al. (arXiv:2606.29039, June 2026).
    - A normalizing-flow posterior for **one** Galactic binary, and for **two** as a
      proof of concept, with label switching broken by ordering f₀.
    - Uniform priors in ±2 μHz boxes, SNR 10–100, T_obs = 1 yr.
  - **Also close:**
    - **Mao, Lee and Edwards (Auckland, arXiv:2512.18290):** flow-matching posteriors for
      one GB-like source with 3 parameters and data gaps, from a WDM-image CNN.
    - **SlotFlow (Houba, Giarda and Speri, arXiv:2511.23228):** amortised inference with
      an unknown number of sources, using factorised per-slot flows and Hungarian
      matching, tested on sinusoids only.
  - **What is ours:**
    - the non-factorised joint posterior of 4 GBs per window;
    - label switching handled inside the flow-matching coupling;
    - Riemannian flow matching on the parameter manifold;
    - broad priors, and one network for the whole band (B);
    - the loss-sensitivity account of the loud-source floor, and its fix. The others see
      the same symptom but attribute it to other causes.
  - **Must do:** the MCMC comparison, and a cut of the scorecard by source separation.
  - **Time-sensitive:** Korsakova's group (NPE, flow matching and diffusion talks) and the
    SlotFlow group ("towards LISA", May 2026) are heading to the same problem.

- **Update, 2026-10-06 (F25).** The B benchmark: 1.57 s per step at 512 wide and 2.57 s
  at 768 (41× XS), memory no issue. The B eval needed draws in chunks, now done. B
  scorecards use `--n_random 50`. Default plan: `B-late` (512), 200k steps, ~3.6
  days.
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

## 2. Setup (`fml` @ dff3ef2; the WDM-era recipe, superseded on 2026-10-12 by F32)

| | XS as trained |
|---|---|
| problem | 4 sources per window, T_obs = 2 yr, `response_points: 16` → a 2 × 64 × 3 (A/E/T) WDM image. Sources are drawn in the central 48 of the 64 bins. f₀ ∈ [0.1, 1.3] mHz, which is 53% of the log band. The SNR prior is 7–1000. |
| flow coordinates | 11 per source: f₀ and A are log-whitened to [−1, 1] over the window (1 unit = 24 bins); log M_c is standardised; the sky lives on S², (ψ, ι) on S², φ₀ on S¹. A `Set` geometry over the 4 sources assigns each draw to its nearest-source permutation. |
| network | `LisaFlow`: MMDiT with hidden 512, 8 blocks, 8 heads, expand 2. It sees 32 y-tokens and 4 x-tokens. **75.6M parameters, the same in XS, S and B.** It has three heads: the velocity, an x₁ prediction (`u_pred`), and a clean-data reconstruction (y). |
| objective | Σᵢ wᵢ Lᵢ / Var(targetᵢ), with w = (1, a, a). a runs a cosine from 1 to 0 over `warmup_frac` × `total_steps` (0.5). t ~ U(0, 1), straight geodesic path, no σ_min. |
| optimiser | global-norm clip at 1, then `optax.contrib.muon` (0.2.8, default width scaling) at lr = 1e-4. Adam, at the same lr, handles the non-matrix parameters. **Constant lr, no schedule**, no weight decay. bf16 compute, fp32 parameters, batch 256, 1 epoch = 1000 steps. |
| the rungs XS / S / B | They differ only in `response_points` (rp), the number of frequency bins of each source's response that jaxgb computes. The window is 4·rp bins, so the image is 2 × 4rp WDM pixels and the network sees 2rp y-tokens. A source's Doppler sideband plus its chirp drift must fit, and that caps f₀. **XS:** rp = 16, a 64-bin window, 32 tokens, f₀ ≤ 1.3 mHz (53% of the log band). **S:** rp = 64, 256 bins, 128 tokens, f₀ ≤ 4.2 mHz (78%). **B:** rp auto-sized to 1024, 4096 bins, 2048 tokens, the full 0.1–12 mHz. Everything else is the same: network, 4 sources, priors, optimiser, 500k steps. |
| cost (A100-80G) | XS takes 38.9 s per epoch, about 23 ms fixed plus 15 ms of network per step. S takes 80 s per epoch. |
| eval | 10 injections at the 0.1…1.0 SNR quantiles of 1024 prior draws (window SNR 121–1848), 1024 flow draws each, RK4 with 32 steps. The Fisher overlay is the Hessian averaged over 32 noise realisations, plus the prior precision. Since 2026-10-06, each injection also gets a **pooled** page (sources stacked, `[draw × source, param]`, one f₀ spike per GB), and the run gets **per-GB** pages for the loudest, median and faintest of the 40 eval sources (draws relabelled onto the source). A log axis spanning less than a factor of 2 (f₀) is drawn linear. |

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
| 13875733, 13875737 | B0: `B-late` and `B-late-768`, 300 steps each into `outputs/bench` | Completed in 10 and 15 min: 1.57 and 2.57 s per step, peak 18.6 and 29.4 GiB (F25). |
| 13875770, 13875840 | B0: scorecard (`--n_random 10`) and eval of the 768 benchmark checkpoint | The scorecard runs at ~78 s per injection. The eval ran out of memory at 1024 draws, now fixed by chunking (F25). |
| **13882121 → 13882126** | B1: `B-late` (512 wide), 200k steps, the last 40k cooling down; five 24 h links chained with `afterany`, the fifth a spare | **Completed** at 08:58 CEST on 10 Oct. It started at 18:46 CEST on 6 Oct; links 1–3 hit the 24 h limit at epochs 56, 112 and 167 (trexgpu05, 04, 04), link 4 ran 168–200, and the spare exited in 57 s. 1.54 s per step, peak 18.6 GiB, no NaN, no queue wait between links. Final 10-epoch flow loss 0.473 (0.498 before the cooldown). The y aux loss jumps from 0.005 to 2.4 the epoch the aux weight reaches 0, then climbs to 1508. XS-late and XS-late-768 did the same (y ended at 190 and 520): once the weight is 0 the aux heads stop training, and sampling uses only the velocity output (F26). |
| 13948963, 13949002 | scorecard (`--n_random 50`) and eval of B-late, queued with `afterok:13882126` | Completed in 43 and 41 min, peak 5.5 GiB each. f₀ is 2.8–4.5 bins wide at every SNR; 1 of 240 sources found; calibration conservative (F26). Downloaded to `outputs/lisa-B-late` (checkpoint 200, verified on the CPU; corner pages; logs). |
| **14263742** (+ 14263743, 14263745) | R1: `XS-pos` (PositionedLisaFlow, 32 periods), 1.2M steps from scratch, the last 200k cooling down; its scorecard and eval chained with `afterok` | **Completed** at 12:06 CEST on 11 Oct (13 h 18 min, trexgpu01); scorecard 4.7 min, eval 13.5 min. Loud floor 0.0067 bins (F31). Checkpoint 1200, corner pages and logs downloaded to `outputs/lisa-XS-pos` and verified. Earlier notes: running since ~22:47 CEST on 10 Oct on trexgpu01 (TREX code f55ab93). 38 s per epoch, peak 2.3 GiB, no NaN. **It lags its baseline:** until epoch ~57 it tracks XS-late (13474448: same recipe, seed and aux phase), then XS-late's flow loss drops by ~0.03 in 15 epochs and R1's does not. At epoch 382 it is 0.524 against 0.476, and the gap is still growing (§6, update of 2026-10-11). |
| 14275317 | R1 mid-run scorecard (epoch 474, constant lr), submitted from here through the stdin shell | Completed in 4.6 min. Loud floor 0.010 bins, so the rule passes half-way; 31 fewer sources found (F30). Copied to `outputs/scorecards/XS-pos_ep474.npz`. |
| **14275499 → 14275503** (+ 14275504, 14275505) | B2: `B-pos` (B-late + `positions: 32`), 200k steps, the last 40k cooling down; five 24 h links with `afterany`, then the scorecard (`--n_random 50`) and eval with `afterok` on the fifth | Submitted 04:15 CEST on 11 Oct (TREX at 3eb92aa, pulled from a bundle). |
| 14275499 | B2, link 1 | **Cancelled** at 03:22 CEST on 12 Oct, after 23 h 11 min (the user, together with an interactive session); links 2–5 never ran. Its checkpoint, at about epoch 56 of 200, is left in `outputs/lisa-B-pos` on TREX. |
| **14326113 → 14326114** (+ 14326115, 14326116) | XS-fd (F32): frequency-domain input, PLE, schedule-free Muon, 1.2M steps; a spare link with `afterany`, then the scorecard and eval with `afterok` | Started 04:22 CEST on 12 Oct, trexgpu03. TREX at 6a0661c, from a bundle; the old head is on branch `trex-pre-fd`. |
| **14326117 → 14326121** (+ 14326122, 14326123) | B-fd (F32): the same recipe on B, 200k steps; five 24 h links with `afterany`, then the scorecard (`--n_random 50`) and eval with `afterok` | Started 04:22 CEST on 12 Oct, trexgpu03. |
| **14326113** (+ 14326114, 14326115, 14326116) | XS-fd: training, then its scorecard and eval | **Completed** at 22:14 CEST on 12 Oct: 17 h 52 min, trexgpu03, 52–53 s per epoch (38–39 s for the WDM nets), no NaN. Final training loss 0.4507, held-out loss of the average 0.4418. The spare link exited at once. Scorecard 4.2 min, eval 14 min. Downloaded to `outputs/lisa-XS-fd`: `checkpoints/1200` (59 files, sizes identical to TREX's, restores on the CPU, all 72.2M parameters finite), `corner/`, `scorecard.npz` and `logs/` (F33). |
| 14401076 | XS-fd scorecard of the training weights y (`--weights training`), TREX at 7adfd3d | Queued at 23:16 CEST on 12 Oct; every A100 was busy. |
| 14401101 | B-fd scorecard of y (`--n_random 50 --weights training`), `afterok` on link 5 | Queued with the B-fd chain. |
| 14326117 → 14326121 | B-fd | **Cancelled by the user** at 02:11 CEST on 13 Oct, at epoch 48 of 200 (training loss 0.558, x's held-out loss 0.542); the later links never ran. The user had scored epoch 45 (14404725, 14404726, eval 14404727). |
| 14408555, 14408558, 14408559, 14408560 | schedule-free Muon comparison A–D (F35), from `~/canna-dev` at 9bdc2d4 | Cancelled at 02:23 CEST on 13 Oct, after 9 epochs of A and B; C and D produced nothing. |
| **14410743** | XS-fd with schedule-free + the SF-NorMuon/ScheduleFree+ fixes, NorMuon rows, 1/L1 weighting, 16 bins per token, ordered sources, reconstruction head (F39, a4da413); `slurm/lisa-local.sbatch` (trains in the node's /tmp, then scorecards x and y and the eval, then copies `outputs/lisa-XS-fd` home) | **Completed** 14:37 CEST on 13 Oct (11 h 30 min with scorecards and eval). Final: training loss 0.739, held out x 0.727 / y 0.736, weight RMS 0.0070, gradient norm 8.8, z − x 0.013 (F40). On TREX in `outputs/lisa-XS-fd`. Locally as `outputs/lisa-XS-sfplus`: checkpoint `flow` and histories complete; `opt_state` (x, z) truncated by a dropped connection; TREX's scorecards and eval not yet fetched. |
| **14410744** | B-fd with the same recipe, 200k steps, scorecards with `--n_random 50` | **Completed** 13:34 CEST on 13 Oct (10 h 27 min with scorecards and eval). Final: training loss 0.669, held out x 0.645 / y 0.654, weight RMS 0.0118, gradient norm 8.4, z − x 0.021. On TREX in `outputs/lisa-B-fd` (beside the cancelled B-fd's `checkpoints/48`); not yet fetched. |
| 14404725, 14404726 (+ 14404727) | B-fd mid-run: scorecards of x and y (`--n_random 50`, `--output_dir outputs/B-fd-ep45`) and the corner eval of x, on checkpoint 45 hard-linked out of the running link 1 | Submitted 00:52 CEST on 13 Oct. Scorecards completed in 46 min each (trexgpu01): loud f₀ 9.5 bins, 3.3× B1's final, but falling with SNR (F38). Copied to `outputs/lisa-B-fd/ep45/`. |

## 4. Findings

### F41 (2026-10-14): the TREX reads — XS's average changes nothing; B matches B1 at a ninth of the cost, against a flat ~3-bin floor

**Fetched** (00:30–01:30 CEST, 14 Oct, once the ControlMaster was back), verified file by file
against TREX (143 files identical):
- XS-sfplus's full checkpoint, its TREX scorecards of x and y (`scorecard_trex.npz`,
  `scorecard_training_trex.npz`) and its corner pages;
- B-sfplus's checkpoint 200, its scorecards of x and y (`--n_random 50`, the 240 sources of
  B1 and B-fd), its corner pages and logs.

Locally they sit in `outputs/lisa-XS-sfplus` and `outputs/lisa-B-sfplus`, with symlinks
`outputs/sfplus/lisa-{XS,B}-fd` for `--output_dir outputs/sfplus`. TREX's XS copy was then
deleted (quota 9.0 → 8.0 GB). B's stays until its float32 scorecard has run.

**XS: the average x does not change F40's verdict.**
- TREX's y scorecard agrees with the local one: 674 against 673 found.
- **x:** 682 of 840 found (y 674, XS-fd 732, R1 740), loud floor 0.0151 bins (12× ideal),
  in68 0.79 when loud.
- **x against y, per source:** 0.99–1.04× in width; x finds 8 more sources and covers better.
- **x against XS-fd, per source:** 1.50–1.62× wider at every SNR (vs R1: 1.41–2.42×).
- **Against the MCMC (x):** loud f₀ 21.7× (y 20.4×, XS-fd 11.4×, R1 6.3×), loud A, sky and
  orientation 7.2×, 8 of 16 missed. Coverage 127/128.
- **Corner page of the loudest source** (SNR 1366): XS-fd resolved the ψ–φ₀ degeneracy as
  clean stripes and A tightly. XS-sfplus smears ψ and φ₀, has a long low-A tail, and a longer
  A–ι ridge. The precision loss is broad, not only in f₀.

**B-sfplus, the same 240 sources as B1 and B-fd:**

| | found (< 1 bin) | < 2 bins | median f₀ width (bins), SNR < 15 / 15–40 / 40–100 / ≥ 100 | slope vs SNR | in68 / in95 / rank, SNR ≥ 15 |
|---|---|---|---|---|---|
| B1 final (WDM, cooled, 85 h) | 1 | 2 | 5.0 / 3.8 / 2.9 / 3.2 | −0.05 | 0.88 / 0.99 / 0.43 |
| B-fd, epoch 45 (F38) | 0 | 0 | 342 / 16.7 / 10.5 / 9.5 | −0.28 | 0.96 / 1.00 / 0.49 |
| **B-sfplus x** (10 h) | 2 | 25 | 4.4 / 3.4 / 3.0 / **3.24** | +0.01 | 0.93 / 0.99 / 0.51 |
| B-sfplus y | 9 | 33 | 7.6 / 4.1 / 3.2 / 2.99 | −0.08 | 0.57 / 0.93 / **0.62** |

- **x matches B1's final model,** per source 0.88 / 0.81 / 0.99 / 1.03× B1's width by SNR band,
  and is better calibrated (rank 0.51 against 0.43). It does that in 10 h of training
  against B1's 85 h, thanks to the 16-bin tokens (F34).
- **The floor is still there:** flat in SNR (slope +0.01) at ~3.2 bins. The rule (< 1 bin)
  fails, as B1's did.
- **y is sharper but biased** (rank 0.62, in68 0.57): not usable as it is.
- **Training diagnostics** follow XS's pattern:
  - weights 0.021 (epoch 8) → 0.0118, and gradient 0.49 → 8.4;
  - z − x bounded at 0.021, x ahead of y by 0.004–0.009 from epoch ~45;
  - a loss step between epochs 60 and 80 (0.720 → 0.699).
- **Corner page of the loudest eval source** (SNR 1668): f₀ is bimodal, with ~20% of the
  draws ~2.5 µHz (~160 bins) from the truth, in λ and β too. A and ι follow the Fisher
  ellipse, wider; ψ and φ₀ are near their prior.

**The bf16 cast of the source embedding** (`lisa_checks/embed_resolution.py`, F38's
check). The f₀ coordinate's resolution after the cast to the bf16 backbone:

| | along the embedding's gradient | worst case | the run's loud floor |
|---|---|---|---|
| XS-sfplus | 4.9×10⁻⁴ bins | 0.026 bins | 0.015 bins |
| **B-sfplus** | 0.094 bins | **2.7 bins** | **3.2 bins** |

On B the floor sits at the cast's worst-case bound, and the f₀ PLE bin is 12 frequency bins
wide (0.19 on XS). Suggestive, not decisive: a float32 scorecard of B's x (14505803, the
same 240 sources) tests it directly.

### F40 (2026-10-13): XS-sfplus — schedule-free now works as a mechanism, but the run regressed on every comparable metric

**Runs.** XS-sfplus (14410743) and B-sfplus (14410744): F39's recipe, both completed with
their scorecards and eval on TREX. The ControlMaster closed during the download (18:30 CEST).
- **Local:** XS's trained weights y and its full histories.
- **Not yet fetched:** XS's x (opt_state was truncated), both runs' TREX scorecards and
  corner pages, and all of B.
- **Scored locally from y:** `lisa_checks/scorecard_weights.py` (840 sources, 5 min on the
  laptop GPU), and `lisa_checks/eval_draws_weights.py` with `compare_mcmc.py --tag
  _sfplus_y` (four windows; `lisa_checks/mcmc_table.py`).
- **Fixed on the way:** `compare_mcmc.py` now relabels the cached Fisher draws onto the
  truth. They kept the old source order, and the loudest-first ordering broke it.

**The schedule-free mechanism is fixed** (per-epoch diagnostics):
- z − x stays bounded at 0.013 RMS from epoch ~75. XS-fd ended at 0.49.
- The average beats the training weights: held-out x − y = −0.005 to −0.009 from epoch 200
  on, growing. XS-fd ended at +0.012.

**But the optimizer drove the weights down.**

| epoch | backbone weight RMS | gradient norm (pre-clip) |
|---|---|---|
| 48 (end of warm start) | 0.027 | 0.29 |
| 200 | 0.015 | 2.4 |
| 1200 | **0.0070** | **8.8** |

- **The clip at 1 binds from epoch ~100 on.** Most of the backbone sits behind
  normalisations, so its gradient grows as 1/|w|.
- **The loop.** The 1/L1 weighting reads the *clipped* gradient. Its L1 rises over the
  warm-start reference, so the step shrinks. The decay was tied to the same weighted lr, so
  the equilibrium weight norm, ≈ 0.2·√(s·lr/2λ), falls with the weighting factor s, and the
  gradient grows further.
- **Without the weighting,** λ = 0.05 at lr 1e-3 balances near 0.020, close to the initial
  0.0255.
- **The paper's warning.** ScheduleFree+ pairs its adaptive step with "fully decoupled" decay
  ∝ γ², whose equilibrium does not depend on γ, and warns that the coupled form "WILL break
  learning rate adaptation". That is what happened. F39's integration was wrong on this
  point.
- **No late loss drop.** The effective (angular) step grows ~4× over the run, the opposite
  of an annealing. In R1 and XS-fd the late loss drop coincided with growing weights
  (F34). That supports reading the drop as their weight growth's implicit lr decay.

**Scorecard of y, the same 840 sources as F33** (`scorecard_training.npz`, matched by
injection and SNR):

| SNR | found: R1 / XS-fd (x) / **XS-sfplus (y)** | f₀ width per source, XS-sfplus / XS-fd [16–84%] | in95 | rank |
|---|---|---|---|---|
| < 15 | 0.53 / 0.50 / **0.35** | 1.67 [1.07–2.98] | 0.92 | 0.49 |
| 15–40 | 0.85 / 0.84 / **0.69** | 1.46 [1.13–2.66] | 0.98 | 0.50 |
| 40–100 | 0.96 / 0.95 / **0.89** | 1.58 [1.17–2.42] | 0.98 | 0.50 |
| ≥ 100 | 0.995 / 0.997 / 0.98 | 1.54 [1.07–2.25] | 0.95 | 0.50 |

- **Loud floor 0.015 bins** (12× ideal), against XS-fd's 0.0103 and R1's 0.0067.
- **673 of 840 found**, against 732 and 740: 67 found only by XS-fd, 8 only by XS-sfplus.
- **Calibration is good** (rank 0.49–0.50, in95 0.92–0.98). The posteriors are honest but
  wider.

**Against the MCMC** (y, four windows, median ratio to the MCMC width):

| | R1 | XS-fd x | XS-fd y | XS-sfplus y |
|---|---|---|---|---|
| loud f₀ (SNR ≥ 500) | 6.3 | 11.4 | 13.8 | **20.4** |
| loud A, sky, orientation | 5.0 | 6.1 | 5.6 | 7.5 |
| f₀ at SNR 50–500 | 2.7 | 3.7 | 3.6 | 6.4 |
| MCMC median inside the 95% | 125/128 | 126/128 | 125/128 | 127/128 |
| sources missed, of 16 | 6 | 6 | 5 | 7 |

The new miss is window 9's SNR-33 source, found by every earlier model. Window 9's SNR-14
source is better than ever (f₀ 15× against 27–45×; sky 1.1–1.7× against 6–10×).

**Why: not separable from one run.** XS-sfplus changed six things against XS-fd (F34, F39).
Three can each plausibly widen everything:
1. **4 tokens instead of 32.** The backbone's work on the data drops ~8×, and close pairs
   share a token.
2. **SNR-ordered labels.** Paths are 38% longer (F34), and faint near-equal-SNR sources swap
   labels inside one posterior.
3. **The optimizer regime above:** weights 3.6× below their initial size, a rising effective
   step, a permanently binding clip.

The other three (NorMuon, the reconstruction head at weight 0.1, and the averaging fixes) are
less likely to cost 1.5× everywhere. The averaging fixes demonstrably helped (x beats y).

**B-sfplus.** Its losses (0.669 training, x 0.645) and diagnostics (weights 0.0118, gradient
8.4, z − x 0.021) show the same pattern. Its scorecards are on TREX, not yet read.

### F38 (2026-10-13): B-fd's last read, epoch 45 of 200 — 10 bins wide, but no longer flat in SNR

**Runs.**
- B-fd's link 1 (14326117) saved checkpoint 45 at 00:34 CEST on 13 Oct, 20.5 h in. It was
  hard-linked into `outputs/B-fd-ep45/lisa-B-fd` on TREX, so the epoch-50 save could not
  replace it under the jobs, at no quota cost.
- Scored there with `--output_dir outputs/B-fd-ep45`:
  - scorecards of the average x (14404725) and of the training weights y (14404726), with
    `--n_random 50`, as B1's: 46 min each, side by side on trexgpu01;
  - the corner eval of x (14404727).
- Locally: `outputs/lisa-B-fd/ep45/`, and `outputs/scorecards/compare_bfd.py 45` (log
  `compare_bfd_ep45.log`).
- **Caveat:** the only B reference is B1's *final, cooled* model (epoch 200). B1 kept one
  checkpoint and was never scored mid-run.

**Loss.** B-fd's training loss minus B1's at the same epoch: +0.053 (epoch 10), +0.050
(20), +0.037 (30), +0.024 (40), **+0.018 (45)**. B-fd falls 3.5× faster per epoch
(−0.0019 against −0.0005 over epochs 40–45), so it would have crossed B1 near epoch 50–55, as
XS-fd crossed R1. The gradient norm is still rising (0.78).

**Scorecard: the same 240 sources as B1's (F26).**

| SNR | n | median f₀ width (bins), B1 final / x / y | × ideal B1 / x / y | in68 B1 / x / y | in95 B1 / x / y | rank B1 / x / y |
|---|---|---|---|---|---|---|
| < 15 | 40 | 5.0 / 342 / 331 | 98 / 6488 / 5073 | 0.90 / 0.78 / 0.70 | 1.00 / 0.95 / 0.93 | 0.49 / 0.53 / 0.53 |
| 15–40 | 38 | 3.8 / 16.7 / 13.5 | 197 / 668 / 595 | 0.84 / 0.95 / 0.92 | 0.97 / 1.00 / 1.00 | 0.38 / 0.48 / 0.49 |
| 40–100 | 47 | 2.9 / 10.5 / 10.5 | 361 / 1263 / 1164 | 0.89 / 0.94 / 0.96 | 0.98 / 1.00 / 1.00 | 0.43 / 0.51 / 0.53 |
| ≥ 100 | 115 | **3.2 / 9.5 / 9.9** | 2148 / 5756 / 5693 | 0.89 / 0.98 / 0.93 | 1.00 / 1.00 / 1.00 | 0.45 / 0.49 / 0.49 |

- **Width.** The loud median is 9.5 bins (95% CI 9.0–10.5), 3.3× B1's final 3.2. No source
  is found (< 1 bin) by x; y has one source under 2 bins, B1 has two. Faint sources
  (SNR < 15) are not localised at all: 340 bins in a 4096-bin window, against B1's 5.
- **x against y.** Per source, y/x = 0.98 [0.71–1.14]. Unlike XS-fd at its end (F33), the
  average does not lag y in f₀ yet.
- **The width now falls with SNR.** The log-log slope of width against SNR (SNR ≥ 15) is
  **−0.28 for x and −0.20 for y, against B1's −0.05**. B1's width was flat in SNR, a
  resolution floor (F26). B-fd's shrinks from 16.7 to 9.5 bins between SNR 15–40 and
  ≥ 100. So at epoch 45 it is still training-limited, not yet against a wall.
- **Calibration.** Unbiased (ranks 0.48–0.53 at every SNR, where B1's sat at 0.38–0.45)
  and conservative (in68 0.93–0.98 above SNR 15), so the width is honest. The
  offsets of the found-and-loud sources are a few bins; the mean is dominated by
  unlocalised faint sources (|offset| up to 2000 bins).
- **In the flow coordinate.** 9.5 bins is 6.2×10⁻³ of x_f (1 unit = 1536 bins on B, F26),
  against B1's 2.1×10⁻³ and XS-fd's final 4.3×10⁻⁴. It is also 0.8 of an f₀ PLE bin
  (2/256 of x_f = 12 bins on B, 0.19 bins on XS). XS-fd ended at 1/20 of a PLE bin. Matching
  that on B would give ~0.6 bins, just inside the rule.

**Weight decay (F34).** B-fd runs with the same weight decay of 1e-6 × lr, effectively
none, so its weights grow as XS-fd's did (18–19×), with an implicit, uneven lr decay. Its
rising gradient norm (0.54 at epoch 6 → 0.78 at 45) fits that. Nothing here separates it
from ordinary training.

**Epoch 48, the last checkpoint.** The user cancelled B-fd at 02:11 CEST on 13 Oct, at
epoch 48, while these jobs ran (F35). Checkpoint 48 (training loss 0.5580, 0.0155 behind B1)
is on TREX and has not been scored: three epochs, −0.004 in loss. Checkpoint 45 is backed up
locally in `outputs/lisa-B-fd/ep45/checkpoints/45`: 35 files, sizes identical to TREX's.

**Reading.** Mid-run, at constant lr, B-fd is 3× wider than B1's cooled final, but its
loss is about to cross B1's, its width depends on SNR as B1's never did, and it is
calibrated. Nothing here failed the rule (< 1 bin at the end), and nothing yet showed it
would pass: it needs another 10× in width over 155 epochs, without the cooldown that gave
the WDM nets their last ×2–3 (F23, F31).
- **A risk for the final rule:** F33's bf16-cast bound on the source embedding, scaled from
  XS to B's coordinate (×64 in bins), is ~0.06 bins typical and ~2 bins worst case. If
  the next B run stalls at 1–3 bins, check that first.

**Eval (corner pages, 14404727, 42 min).** `outputs/lisa-B-fd/ep45/corner/` (23 pages).
- **Loudest eval source (SNR 1668):** f₀ is centred. But λ has three or four separate modes,
  A has a low-amplitude tail, and ψ and φ₀ are close to their prior, all against a tight
  Fisher ellipse.
- **Median source (SNR 61):** f₀, sky and A sit on the Fisher ellipse, with a few small
  outlier islands. ψ and φ₀ are broad.
- So the loud sources are where the run is still immature, in every coordinate, not only
  f₀. That fits a mid-run model rather than a defect.

### F39 (2026-10-13): SF-NorMuon's and ScheduleFree+'s fixes, integrated (renumbered from F38, which a parallel session also used)

`src/canna/lisa/schedule_free.py` replaces `optax.contrib.schedule_free`. The configs
(XS-fd, B-fd) use it with:
- **Weight decay on z** (SF-NorMuon): z ← z − η λ z + base step, λ = 0.05, each parameter at
  its group's lr (Muon's or AdamW's), matrices and embedding tables only. The base optimizer
  no longer decays. Without `--schedule_free`, decay stays optax's decoupled one on the
  weights.
- **Inner momentum** (both papers): Muon momentum 0.8 (SF-NorMuon's), AdamW b1 0.9 (was 0).
- **Averaging weights t^r lr², r = 1** (ScheduleFree+): late iterates weigh more.
- **Warm-started averaging** (ScheduleFree+): x follows z for the first 2 × warmup_steps.
- **Interpolation β from 0.9 to 0.965**, log-linear in t over total_steps (ScheduleFree+).
- **x kept in the state**, not rebuilt from y and z: a varying β needs it, and it removes
  F34's float32 reconstruction error.

Not taken over (corrected 2026-10-13 after reading ScheduleFree+ in full):
- **The Polyak step** (its Eq. 12, Algorithm 1). It is the lr-free mechanism and needs the
  loss minimum f*; Algorithm 1 drops it, i.e. takes f* = 0. Flow matching has a large
  irreducible loss (~0.4 on XS), so f* = 0 would scale the steps with the whole loss, not
  the excess.
- **AdamC's fully decoupled decay** (γ_t² λ z, λ 5–50; Eq. 14). The paper calls it
  "necessary when using the Polyak step size". With a set lr, the SF-NorMuon form used here
  (γ λ z) differs from it only through the warmup.
- **NorMuon's row-wise normalisation**: a different base optimizer, not a fix (SF-NorMuon
  credits it with ~12% faster convergence).
- **Inverse gradient-L1 step weighting** (Eq. 10): left out by mistake. It is *not* tied to
  the lr-free step: the paper also runs it with a grid-searched lr ("ScheduleFree 1/L1",
  Fig. 8). The step size goes as 1/EMA‖g‖₁ (EMA 0.9), to counter gradient-norm drift. That
  drift shows in our runs: the gradient norm rose from 0.30 to 0.59 late in XS-fd, and from
  0.63 to 0.80 by epoch 48 of B-fd.

**Checks.**
- `tests/lisa/test_schedule_free.py`:
  - with the fixes off (r = 0, fixed β, no decay, no warm start) it matches
    `optax.contrib.schedule_free` step for step, in y and in x;
  - decay on z holds a constantly pushed z at u/(ηλ), where it would otherwise grow
    linearly;
  - x follows z until the warm start;
  - r = 1 brings x to the late iterates faster;
  - β runs monotonically from its start to its end.
- The full suite passes (844).
- A 400-step laptop run (XS-fd, warmup 50): z − x stays 0 for the 100 warm-start steps, then
  opens; the backbone weight RMS holds at 0.019.

Checkpoints now carry x as well: one more copy of the weights, ~0.3 GB on XS.

### F37 (2026-10-13): the literature on schedule-free failures — F36's runaway is documented for LLMs; no flow-matching reports

Search of 13 Oct. No paper I found trains a diffusion or flow-matching model with a
schedule-free optimizer, so there is no report of issues specific to flow matching. The
failure of F36 is documented for language models:

- **SF-NorMuon** (Apte, Deshpande, Kumar, Chakrabarti, Kim, arXiv:2605.23061, May/Sep 2026).
  Schedule-free with a normalised-Muon base. In their words, "for SF methods, the validation
  loss at X diverges without weight decaying specifically at Z".
  - SF-NorMuon with no decay "diverges faster" than SF-AdamW.
  - Decay at Y, the original schedule-free recommendation, is unstable too.
  - **Mechanism, as in F36:** Z runs at a constant lr and its loss "often diverges". X
    averages all past Z, so Z "eventually contaminates" X.
  - **Fix:** decay on Z, Z ← Z − ηλZ − η̂P, as part of the base dynamics, not as an ℓ2 term.
    It bounds every iterate, and |Z| converges to a steady state.
  - **Settings (125M LM):** η = 0.01, λ = 0.05, interpolation β1 = 0.9, Muon momentum 0.8.
  - **Ablation:** with Muon's momentum removed (μ = 0), the final loss is worse (3.14 →
    3.28). So keeping Muon's own momentum inside schedule-free, as XS-fd did, is right.
- **ScheduleFree+** (Defazio, arXiv:2605.19095, May 2026). Fixes needed to scale
  schedule-free:
  - inner momentum in the base (large-batch divergence without it);
  - fully decoupled weight decay;
  - warm-started averaging (no averaging for ~2× the warmup);
  - interpolation β annealed from 0.9 to 0.965;
  - **averaging weights r = 1 for long runs.** The default, uniform after the warmup,
    "underweights later iterates", which is F33's lag after the late loss drop.
  - Evaluate x, the average, not y.
- **AMUSE** (Kim et al., arXiv:2605.22432): a fixed interpolation β trades early speed
  against late stability; it moves β towards 1 over time.
- **Diffusion, weight growth and averaging** (from prior knowledge, not this search):
  - Karras et al., "Analyzing and Improving the Training Dynamics of Diffusion Models"
    (EDM2, arXiv:2312.02696), found uncontrolled growth of weight magnitudes during
    diffusion training. With normalisations, the growth acts as an uncontrolled, uneven
    decay of the effective lr. They removed it by forcing the weights to unit norm.
  - The same paper fits a power-function EMA of the training weights after the fact. The
    diffusion literature evaluates EMAs of the training iterate, not schedule-free
    averages.

**Consequences for our code.**
- **Weight decay at y is the wrong place.** optax's `schedule_free` passes y to the base
  optimizer, so `optax.contrib.muon`'s decay acts at y. Even with λ = 0.05 that is the
  placement SF-NorMuon finds unstable. Decay must act on z: a schedule-free wrapper of our
  own (~40 lines, optax's plus the −ηλz term), or decay off in the base with z decayed in
  the wrapper.
- **Averaging weights.** For long runs, take ScheduleFree+'s r = 1 and warm-started
  averaging, or AMUSE's annealed β. They address the lag after the late loss drop
  (F33, F36).
- **Alternative with the most precedent in diffusion:** constant-lr Muon with decay, and an
  EMA (or post-hoc EMA) of the training weights.

### F36 (2026-10-13): why schedule-free failed on XS-fd — its fast sequence z ran away, and only the training point y stayed good

Diagnosis of XS-fd's final schedule-free state (`lisa_checks/sf_anatomy.py`,
`lisa_checks/sf_line.py`, run on 7adfd3d, the code that trained it; 8 × 256 common samples).

**The three weight sets.**
- **z is about twice the size of x.** Over the backbone, |z|/|x| = 1.92 and cos(x, z) = 0.80.
  In the MLP input weights, x's RMS is 0.46–0.49 and z's is 0.88–1.01; at initialisation it
  was 0.0255.
- **x is not a denoised z.** If |z| grows in proportion to t, the uniform average sits at
  half its final norm, so x is mostly a radially shrunk z, plus some averaging of directions.
- **z has run off.** It has moved ~1.0 RMS per element in 1.2M steps. A pure random walk of
  Muon's steps (2×10⁻⁴) would move 0.22, so z carries a persistent drift.

**Loss along the line from x to z:**

| point | flow loss |
|---|---|
| x (α = 0) | 0.467 |
| α = 0.05 | 0.459 |
| **y (α = 0.111)** | **0.4575** |
| α = 0.2 | 0.574 |
| α = 0.3 | 85 |
| α = 0.5 | 1.9×10⁹ |
| z (α = 1) | 2.7×10²¹ |

- **Only a thin neighbourhood of y is a good network,** and x sits on the steep side of it.
- **It is not a per-layer scale lag.** x rescaled layer by layer to y's norms scores 0.519,
  and y rescaled to x's scores 0.522.

**The mechanism.**
1. Schedule-free takes the gradient at y = 0.9x + 0.1z, and the base optimizer steps z. Late
   in the run x hardly moves (c_t ≈ 10⁻⁶), so y moves by 0.1 of z's step. For y to follow
   the gradient, z must travel ~10× as far.
2. Muon's step has a fixed size (RMS 0.2·lr per element) whatever the gradient: Newton–Schulz
   normalises its magnitude away. So z never slows down near convergence, unlike with SGD.
3. With no restoring force (weight decay 1e-9 per step, F34), z drifts without bound and
   leaves the basin.
4. Schedule-free's average assumes the segment from x to z lies in one basin, as in a convex
   problem. Here it crosses a cliff, so the average cannot beat y: XS-fd's x trails y by
   0.012 (F33), and the late loss drop (F10) leaves x carrying the pre-drop iterates on top.

**Consequences.**
- **Weight decay ~0.05 is necessary** (F35: ~40% slower growth at 9k steps). It may not be
  sufficient: in F35, z drifted from x at the same rate with or without decay.
- **Unbounded steps inside schedule-free.** The interpolation multiplies them by 10 in z.
  Whether dropping Muon's own momentum helps (F35's variant C) is untested.
- **The robust option** keeps Muon and evaluates something on y's trajectory: an EMA of y, or
  a short cooldown of y. Those weights never leave the region where the gradients are taken.

### F35 (2026-10-13): schedule-free Muon comparison, stopped at 9k steps — weight decay slows the growth; z drifts from x regardless

**Runs.**
- Four 150k-step XS variants (`slurm/lisa-compare.sbatch`), all with 16 bins per token, the
  ordered sources and the reconstruction head (b9bb3ef, 9bdc2d4).
- **Cancelled at 02:23 CEST on 13 Oct**: there was no more GPU time that day.
  - A (14408555, schedule-free, Muon momentum 0.95, decay 1e-6, XS-fd's settings) and
    B (14408558, the same with decay 0.05) ran 9 epochs (9k steps, the warmup ending at 3k)
    at 31–32 s per epoch.
  - C (no Muon momentum) never finished an epoch, and D (no schedule-free) never started.
- Logs in `outputs/compare/{A,B}`. No scorecards: the checkpoints lived in the nodes' /tmp.

**Diagnostics on the fixed batch** (epoch 9; growth per 1k steps over epochs 5–9):

| | A (decay 1e-6) | B (decay 0.05) |
|---|---|---|
| held-out loss, averaged weights x | 0.9660 | 0.9655 |
| held-out loss, training weights y | 0.9684 | 0.9663 |
| backbone weight RMS (initial 0.0192) | 0.0287 (+50%) | 0.0251 (+31%) |
| its growth per 1k steps | 0.0013, steady | 0.0008, slowing |
| z − x RMS | 0.0214 | 0.0203 |

**Preliminary readings (9k steps, one seed).**
- **Decay 0.05 does what F34 predicted.** Weight growth is ~40% slower and still slowing,
  at no cost in loss so far. A grows at an undiminished rate, on the path to XS-fd's 19×.
- **z drifts away from x at the same rate either way** (~0.002 RMS per 1k steps). After
  9k steps it is already 0.7–0.8 of the weights' own RMS. Decay does not hold z close to its
  average; only the lr or the averaging can. This is the part of the interaction that 150k
  steps were meant to follow.
- **Early on, the average is the better model,** as schedule-free intends: x's held-out
  loss is 0.001–0.002 below y's from epoch 4. XS-fd's reversal (x worse by 0.012) belongs to
  late training, which this test never reached.
- **Not answered:** whether Muon's own momentum hurts inside schedule-free (C), and whether
  schedule-free beats plain constant-lr Muon at all (D).

**What this supports.** Run the next model with weight decay ~0.05. Keep logging x and y,
the weight RMS and z − x (now on by default), and score both x and y. The late-phase
question (x lagging after the loss drop) can only be settled by a full-length run.

### F34 (2026-10-13): schedule-free Muon ran with no weight decay; Muon's cost is Newton–Schulz in float32; sources now ordered

**Training weights y (scorecard 14401076, all 840 sources).**

| | loud floor (bins) | loud in68 | loud rank | detection |
|---|---|---|---|---|
| y (training weights) | 0.012 | 0.74 | 0.48 | 731 / 840 |
| x (average) | 0.0103 | 0.69 | 0.41 | 732 / 840 |

- y is unbiased but wider than x. The averaging does not hide a sharper model, so neither
  weight set is a converged one.

**The schedule-free integration (`lisa_checks/sf_precision.py`, the XS-fd checkpoint).**
- **Mechanics are correct.**
  - Parameters and z stay float32.
  - The only float64 in the update is in scalars: the Newton–Schulz coefficients and the
    bias correction.
  - The wrapper costs 0.7 ms per step over Muon alone.
- **float32 averaging error is small.** Replaying optax's recursion over 1.2M steps,
  float32 lands 1% of x's movement away from float64, with a +0.8% rescale. Late on,
  1 − c_t (c_t = 8.4×10⁻⁷) rounds to a value that stays fixed for ~10⁵ steps, which is
  where that comes from. Minor.
- **The real problem is weight decay.** 1e-6 is multiplied by the lr in optax (as in PyTorch
  AdamW), so the decay was 1e-9 per step, effectively zero. Muon's steps have a fixed size,
  2×10⁻⁴ RMS whatever the gradient.
  - **Weights:** the backbone grew **18–19×** from its initial RMS (0.0255 → 0.46–0.49).
    R1, with Adam at 1e-4, grew 3–4×.
  - **Spread of z:** z sits 0.49 RMS from its own average x, more than |x| = 0.37.
- **Consequences.**
  - Most of the backbone sits behind normalisations, so the effective step falls as the
    weights grow: an implicit lr decay of up to ~19×, varying by layer.
  - Schedule-free averages iterates spread over a region larger than the weights
    themselves.
- **Schedule-free and Muon both rely on weight decay** (AdamW 0.05–0.1 in the schedule-free
  experiments; 0.1 with the RMS-matched lr in Moonshot's Muon). Run without it, this is not
  the regime where "no schedule needed" was shown.
- **Fix for the next run (not yet set in the configs):** weight decay of order 1e-2 to 1e-1
  in optax's convention. The equilibrium weight RMS is about 0.2·√(lr / 2λ): 0.045 at lr 1e-3
  and λ = 0.01, 0.014 at λ = 0.1.

**Where the step time goes.** On the laptop GPU (XS, batch 256, 50-step loops):

| part | time | share |
|---|---|---|
| batch generation | 1.3 ms | 0.2% |
| forward + backward | 344 ms | 42% |
| optimizer, schedule-free Muon | 326 ms | 40% |
| optimizer, Muon alone | 325 ms | 40% |
| optimizer, AdamW alone | 37 ms | 4.6% |

- Muon's cost is Newton–Schulz: ~1 TFLOP per step on 71M parameters, about a quarter of the
  network's own FLOPs. optax runs it in the dtype of the momentum estimate, which follows
  the float32 gradient, even with the momentum stored in bf16.
- That cost is fixed per step: parameters × width, not batch × tokens. With only 256 × 36
  token rows on XS it shows (+37%); with 2048 tokens on B it does not (+5%).
- `newton_schulz_dtype: bfloat16` (b9bb3ef, the default) casts Muon's gradients first, as
  the reference Muon does.

**A100 timings** (jobs 14405794 and 14405795; `lisa_checks/bench_step.py`, logs in
`~/canna-dev/.slurm-logs` on TREX). XS at batch 256, 32 tokens:

| part | time | share |
|---|---|---|
| batch generation | 0.6 ms | 1.5% |
| forward | 6.6 ms | 15% |
| forward + backward | 26.5 ms | 61% |
| optimizer, schedule-free Muon, NS bf16 | **12.1 ms** | 28% |
| optimizer, schedule-free Muon, NS float32 | 21.2 ms | 49% |
| optimizer, Muon alone, NS float32 | 20.8 ms | 48% |
| optimizer, AdamW alone | 1.3 ms | 3% |
| **full step, NS bf16** | **43.4 ms** | |

- float32 NS reproduces XS-fd's 52 ms per step: 43.4 − 12.1 + 21.2 = 52.5.
- bf16 NS saves 9 ms of it. What remains is Muon's own ~12 ms against R1's 38 ms Adam
  step.
- On B the optimizer is under 1% of the step (forward + backward 1550 ms of 1592).

**Fewer tokens** (A100, bf16 NS, full step):

| | patch 2 | patch 4 | patch 8 | patch 16 |
|---|---|---|---|---|
| XS (32 / 16 / 8 / 4 tokens) | 43.4 ms | 36.7 ms | 33.3 ms | 31.1 ms |
| B (2048 / – / 512 / 256 tokens) | 1592 ms | – | **325 ms (4.9×)** | **177 ms (9.0×)** |

- XS gains little: its fixed costs (the optimizer and the batch) dominate.
- On B, patch 8 would make 200k steps ~18 h instead of ~88 h, and patch 16 ~10 h.

**Ordered sources (b9bb3ef).**
- `sample_physical` lists each window's sources loudest first, by each source's own SNR,
  so training targets, prior draws and injections are ordered alike.
- The geometry is one source's `Product`, applied per slot. The `Set` geometry and its
  assignment code are gone.
- LisaFlow gives each slot a learned embedding, and has a second head that reconstructs the
  parameters x1. Its loss is the per-slot squared ambient distance, at a constant weight
  `aux_weight: 0.1`.
- Scoring keeps `match_sources`, so scorecards compare with earlier ones and with the MCMC.
- **Cost of fixed labels: transport paths ~38% longer.** Mean squared path length per
  source, in flow coordinates (4096 XS windows):

| pairing | total | f₀ part | amplitude part |
|---|---|---|---|
| `Set`, nearest permutation (before) | 8.73 | 0.58 | 0.58 |
| SNR order | 12.04 | 0.67 | 0.29 |
| f₀ order | 12.04 | 0.26 | 0.66 |
| random labels | 12.44 | 0.66 | 0.66 |

  Most of the excess is in the angles, which no ordering shortens. SNR order shortens the
  amplitude paths, f₀ order the f₀ paths. Flow losses are therefore not comparable with
  earlier runs'. A laptop smoke run starts at 1.08 against 0.78.
- **Label ambiguity.** Near-equal SNRs swap labels within one posterior (SNR is known to
  ±1). f₀ is known to ~0.01 bins, so f₀ order would rarely swap.

### F33 (2026-10-12): XS-fd — R1's level below SNR 100, a 1.5× wider loud floor, and a lagging schedule-free average

**Runs.**
- XS-fd (14326113, F32's recipe) ran 1.2M steps: 17 h 52 min, no NaN, gradient norm
  0.2–0.6, so the clip at 1 never bound.
- Scorecard 14326115 and eval 14326116 scored the averaged weights x.
- Locally:
  - eval draws for x and for the training weights y (`eval_draws.py ... --weights
    training`, laptop GPU);
  - the MCMC comparison on windows 9, 5, 2 and 7 (`compare_mcmc.py --flow ... --tag _fd` and
    `_fd_y`, logs `outputs/mcmc/compare_fd*.log`);
  - the per-source comparison (`outputs/scorecards/compare_fd.py`, log `compare_fd.log`);
  - the flow loss of x, y and R1 on 32 × 256 identical training samples
    (`outputs/scorecards/common_loss.py`; R1 from a worktree at `wdm-r1`).

**Scorecard against R1 (the same 840 sources).**

| SNR | n | found, 512-cool / R1 / fd | f₀ width (bins), R1 → fd, found by both | per-source fd/R1 [16–84%] | × ideal R1 → fd | in68 R1 → fd | in95 R1 → fd | rank R1 → fd |
|---|---|---|---|---|---|---|---|---|
| < 15 | 148 | 0.50 / 0.53 / 0.50 | 0.090 → 0.075 | 0.93 [0.75–1.35] | 1.8 → 1.6 | 0.76 → 0.75 | 0.97 → 0.97 | 0.52 → 0.51 |
| 15–40 | 157 | 0.85 / 0.85 / 0.84 | 0.044 → 0.043 | 1.00 [0.73–1.38] | 1.9 → 2.0 | 0.78 → 0.76 | 0.97 → 0.98 | 0.51 → 0.47 |
| 40–100 | 151 | 0.95 / 0.96 / 0.95 | 0.016 → 0.017 | 1.07 [0.82–1.40] | 2.0 → 1.9 | 0.77 → 0.70 | 0.97 → 0.94 | 0.53 → 0.46 |
| ≥ 100 | 384 | 0.995 / 0.995 / 0.997 | **0.0067 → 0.0102** | **1.54 [0.98–2.27]** | **4.2 → 7.5** | 0.71 → 0.69 | 0.96 → 0.95 | 0.50 → 0.41 |

- **Loud floor 0.0103 bins (95% CI 0.0096–0.0110).**
  - That is 1.54× R1 (CI 1.46–1.63) and 0.26× the same WDM net without features
    (XS-late-cool, 0.039).
  - By SNR band (100–200 / 200–400 / 400–800 / ≥ 800) it is 0.0127 / 0.0107 / 0.0094 /
    0.0078 bins, against R1's 0.0100 / 0.0063 / 0.0048 / 0.0037. So it falls ×1.6 over
    the range against R1's ×2.7: flatter, more like a precision limit.
- **Everything else is R1's level.**
  - Detection is 732 of 840 against 740, with 10 found only by fd and 18 only by R1.
  - in95 is 0.95–0.98.
  - There is no bias in bins (+0.0007 ± 0.0054, window profile ≤ 0.010). But the loud rank
    is 0.41: the truth sits slightly low in the draws, a few tenths of a width.
- **The scorecard part of §6's rule fails** (floor ≤ 0.0067, detection ≥ R1).

**Against the MCMC (four windows, 128 source–parameter pairs).**

| | R1 | fd, x | fd, y |
|---|---|---|---|
| loud f₀ (SNR ≥ 500), median × MCMC | 6.3 | 11.4 | 13.8 |
| loud A, sky, orientation, median × MCMC | 5.0 | 6.1 | 5.6 |
| MCMC median inside the flow's 95% / 68% | 125 / 108 | 126 / 109 | 125 / 106 |
| sources missed (> 50% of draws > 1 bin off) | 6 | 6 | 5 |

- A, sky and orientation stay **3–9×** with fp32 inputs and fp32 PLE coordinates. **F32's
  hypothesis, that bf16 rounding of the data set this limit, is refuted.** Something
  after the embeddings sets it: the bf16 backbone, the loss's insensitivity to it (F6), or
  training.
- R1's biased SNR-235 source (W1 3.1 and 7.8 σ in A and longitude, F31) is inside the 95%
  now. The new outlier is the SNR-1366 source's sin ι.
- The crowded SNR-64 source (1.1 bins from SNR 819) is worse: 27× in f₀, 22–61× elsewhere.
- The same masked sources are missed, so masking is untouched, as expected.

**Schedule-free's average lags the training weights.**

Flow loss on identical samples (8192 each, paired):

| | flow loss | paired difference |
|---|---|---|
| x (evaluated) | 0.4623 | x − y = **+0.0118 ± 0.0004** |
| y (training) | 0.4504 | y − R1 = +0.0116 ± 0.0006 |
| R1 | 0.4389 | x − R1 = +0.0234 ± 0.0006 |

- **Why x lags.** With a constant lr, schedule-free's x is the uniform average of z over
  the whole run. Both runs show the late loss drop of F10, at 75–90% of the run: R1 at
  epochs 850–900 (−0.034), XS-fd at 950–1100 (−0.042). So x still carries the pre-drop
  weights.
- **The held-out loss of x shows it.** It sat 0.014 below the training median before the
  drop, 0.009 above it at epoch 1000, and was catching up at the end (−0.022 over the last
  100 epochs).
- **Training curve.** XS-fd's training loss was 0.010 below R1's from epoch 50 to 850 (the
  constant-lr phase of both). Its drop came 100–150 epochs later than R1's. At the end it
  was still falling at −0.058 per doubling of steps.
- **The y weights are not sharper at the loud end** (13.8× against x's 11.4× in f₀). So the
  averaging lag explains the loss gap, not the floor.
- **What R1 had instead.** R1's cooldown (as in C1, F23) took its floor from 0.010 bins at
  mid-run, constant lr (F30), to 0.0067. XS-fd's x and y are both constant-lr models, one
  averaged badly and one not averaged at all. The y scorecard (14401076) will show this on
  all 840 sources.

**Not the embedding cast.** The bf16 cast of the trained source embedding limits x_f, read
along the embedding's gradient, to 1.0×10⁻³ bins. That is ten times finer than the floor.
The worst case, all of the error along the wrong direction, is 0.03 bins. A, sky and
orientation come out at 1–2×10⁻⁴ in coordinate units.

**Cost.** 52 s per epoch against 38 s for the WDM nets (+37%). It is mostly Muon's
Newton–Schulz on the 71M backbone parameters, which XS's small network does not amortise.
B-fd costs 1614 s per epoch against B1's 1536 (+5%).

**B-fd at epoch 40** (14326117): training loss 0.571 against B1's 0.547 at the same epoch.
The gap is closing (0.053 at epoch 10, 0.037 at 30, 0.024 at 40), as XS-fd's did before it
overtook R1. The gradient norm is rising (0.63 → 0.77).

### F32 (2026-10-12): Muon never trained the backbone, bf16 rounded the loud data; a new recipe for XS and B

**Muon routing.**
- `optax.contrib.muon` (0.2.8) with no dimension numbers takes exactly the 2-D arrays.
- `MMDiT` stacks its blocks with `filter_vmap`, so every block weight is
  `(blocks, out, in)` and went to Adam.
- On XS-pos that is **67.1M of 75.7M parameters (89%) on Adam at lr 1e-4**, in every run
  from 11000056 to R1 and B2.
- Muon got the embeddings and heads, where it is usually not used. It also got the
  stacked `(8, D)` biases of the blocks, which it orthogonalised across blocks.
- optax reads a weight as `x W`, `(in, out)`, while equinox stores `(out, in)`. So Jordan's
  width scaling was applied to the wrong side of the 2-D weights it did take.
- So §2's "Muon at lr 1e-4" and F9's small steps describe Adam.

**bf16 rounding of the data.** `LisaFlow` cast `y = arcsinh(WDM)` to bf16 before its
first layer. The round-trip error `sinh(bf16(y)) − sinh(y)` over 2048 XS prior windows, in
noise σ:

| pixel | share | rms error | max |
|---|---|---|---|
| < 4σ | 74% | ≈ 0 | 0.03 |
| 4–30σ | 16% | 0.07 | 0.23 |
| 30–100σ | 7.1% | 0.53 | 1.6 |
| 100–300σ | 2.5% | 1.5 | 4.6 |
| > 300σ | 0.2% | 3.6 | 10.6 |

F26 ruled out the bf16 cell of the source's f₀ coordinate, not this one. It is a candidate
for the loud sources' A, sky and orientation widths, 3–9× the MCMC's (F31). The
frequency-domain stream reaches ~735σ per part on XS; T stays below ~5σ.

**The recipe (6a0661c; the WDM code is tagged `wdm-r1`).**
- **Data.** `preprocess` returns the whitened rfft window, real and imaginary parts of A, E
  and T, at unit noise variance per part. The window keeps its WDM-era geometry, so priors,
  windows and scorecards compare one to one. Two bins make a token: 32 on XS and 2048 on
  B, as before.
- **PLE** (Gorishniy et al. 2022, arXiv:2203.05556) for every scalar the network takes in,
  each followed by one linear map, all in fp32:
  - data: 64 bins per value, uniform in arcsinh out to 1000σ, so 0.24σ wide near zero and
    ~70σ at 300σ (quantile edges would put everything above ~80σ in one bin);
  - source coordinates: 64 uniform bins, f₀ 256;
  - window frequency: 64 bins in log f. The old embedding fed f in Hz to a sinusoid, so
    its angles spanned 1e-4–0.08 rad.

  The y tokens get a learned position each. The velocity head and the conditioning run in
  fp32, the MMDiT in bf16.
- **Optimiser.** Schedule-free (Defazio et al. 2024, arXiv:2405.15682) around optax's
  Muon:
  - Muon keeps its own momentum inside the wrapper, as SF-Muon in AMUSE (arXiv:2605.22432,
    eq. 4);
  - Muon: lr 1e-3 with `consistent_rms=0.2`, momentum in bf16;
  - AdamW: b1 0 and b2 0.95, lr 3e-4, for the encodings, the token positions, the last
    head layer and 1-D params;
  - weight decay 1e-6, a 2% linear warmup, clip at 1.

  q/k/v, SwiGLU value/gate and adaLN shift/scale/gate are stacked maps, orthogonalised one
  by one, and the blocks have no biases. **Muon now takes 71.3M of 72.2M (98.7%)**,
  asserted at startup.
- **Removed:**
  - the aux heads (x₁ prediction, clean-signal reconstruction), `warmup_frac` and the
    Welford reweighting;
  - the cooldown and `save_opt_state`;
  - `PositionedLisaFlow` (the user's call).

  The flow loss is the plain velocity MSE.
- **Checkpoints and logs.** Checkpoints keep the optimiser state, since the evaluated
  weights are the schedule-free average, rebuilt from z: ~610 MB on XS. `load_trained`
  returns that average. Each epoch logs it on a fixed held-out batch (`heldout=`), next to
  the training loss at y and the gradient norm.
- **Checks.**
  - The full test suite passes on the CPU.
  - A 300-step bf16 run on the laptop GPU is clean: peak 2.0 GiB, gradient norm ~0.35, so
    the clip is mostly idle.
  - On the CPU the averaged weights move off y as they should.

**What is confounded.** One run per rung changes the data, the embedding, the optimiser
(Adam to Muon) and the schedule, and drops three features at once. A miss will not say
which change caused it. The held-out loss of x against the training loss at y, the
gradient norm and the routing printout are the cheapest diagnostics.

### F31 (2026-10-11): R1 final — the floor drops to 4× ideal with near-nominal coverage, and detection recovers

**Runs.**
- R1 training (14263742) completed at 12:06 CEST on 11 Oct: 13 h 18 min, trexgpu01,
  38 s per epoch, no NaN.
- Scorecard 14263743 (4.7 min) and eval 14263745 (13.5 min).
- Downloaded to `outputs/lisa-XS-pos`:
  - `checkpoints/1200`: 776 MB, file list and sizes identical to TREX's, restores on the
    laptop as a `PositionedLisaFlow` at epoch 1200;
  - `corner/` (23 pages, 140 MB), `losses.pdf` and the logs;
  - `eval_draws.npz`, regenerated on the laptop GPU with eval.py's keys
    (`outputs/lisa-XS/eval-tools/eval_draws.py`). The script reproduces the stored 768-cool
    draws: median per-draw f₀ difference 0.004 bins, width ratio 1.0001.
- Comparison: `outputs/scorecards/compare_r1.py`, against XS-late-cool (the same recipe
  without the features, the same 840 sources).

| SNR | n | found, 512-cool → R1 (768-cool) | f₀ width (bins), found by both | per-source ratio R1/512-cool [16–84%] | × ideal (R1) | in68 | in95 | rank |
|---|---|---|---|---|---|---|---|---|
| < 15 | 148 | 0.50 → **0.53** (0.54) | 0.118 → 0.090 | 0.74 [0.58–0.94] | 1.8 | 0.76 → 0.76 | 0.97 → 0.97 | 0.52 |
| 15–40 | 157 | 0.85 → 0.85 (0.86) | 0.077 → 0.044 | 0.56 [0.40–0.74] | 1.9 | 0.92 → 0.78 | 1.00 → 0.97 | 0.51 |
| 40–100 | 151 | 0.95 → 0.96 (0.97) | 0.043 → 0.016 | 0.40 [0.27–0.65] | 2.0 | 0.93 → 0.77 | 0.98 → 0.97 | 0.53 |
| ≥ 100 | 384 | 0.995 → 0.995 (0.995) | **0.039 → 0.0067** | **0.19 [0.10–0.46]** | **4.2** | 0.93 → 0.71 | 0.99 → 0.96 | 0.50 |

- **The rule passes, by a wide margin.** The loud floor is **0.0067 bins (95% CI
  0.0061–0.0076)**, 2.8×10⁻⁴ units of x_f. The per-source ratio is 0.189 (95% CI
  0.170–0.206): 5.3× narrower than the same recipe without the features, and 5.5× narrower
  than the 768 final in median width.
- **The floor follows SNR:** 0.0100 / 0.0063 / 0.0048 / 0.0037 bins at SNR 100–200 /
  200–400 / 400–800 / ≥ 800. That still falls more slowly than 1/SNR (×2.7 over a factor
  ~6.7 in SNR), so a residual floor remains, now at 4× ideal.
- **Coverage is near nominal,** in68 0.71–0.78 and in95 0.96–0.97 in every band. The baseline
  over-covered (in68 0.93).
- **No bias:** +0.0012 ± 0.0017 bins, and the window-position profile at most 0.002.
- **Detection recovered:** net +7 of 840 against 512-cool (17 against 10). That is close to
  the 768 final (0.54 / 0.86 / 0.97 / 0.995), at 512 width and the 512's cost (13.3 h
  against 10.9 + 2.2).
- **Loss.** The 10-epoch mean is 0.4403 at the end, against 0.4352 for 512-cool. The lag was
  0.048 at epoch ~380, 0.008 at epoch 1000 (0.4634 against 0.4557) and 0.005 after the
  cooldown. R1 caught up during the constant-lr phase.
- **F30's predictions held:** floor 0.004–0.008 bins (0.0067); detection within ~10 sources
  of 512-cool (it is +7); in68 0.68–0.80 when loud (0.71).

**Crowding with the floor gone (F29's cut, `paper/scripts/tables.py`).**
- At SNR ≥ 40, found sources are **0.0062 bins** wide beyond 5 bins of a neighbour, 0.017 at
  2–5 bins and 0.028 within 2 bins. Crowding now sets the remaining width; for the baselines
  the floor hid it (0.036 / 0.046 / 0.057).
- **Detection next to louder neighbours does not improve:** 63% of SNR 15–40 sources within 5
  bins of a >3× louder one are found, as for the 768 final (59% for 512-cool). So F28's
  reading (the loud source's imprecision masks its neighbour) is not supported. The misses
  next to loud sources have another cause, perhaps how slots are assigned or a coarse
  search near a dominant peak.

**Against the MCMC (the four windows of F28; `compare_mcmc.py --flow
outputs/lisa-XS-pos/eval_draws.npz --tag _r1`, log `outputs/mcmc/compare_r1.log`; table
`tab:mcmc-r1`).**
- **f₀ is fixed.**
  - Loud sources go from 38–84 to **5–12× the MCMC width**: 7.3, 5.3, 5.0 and 12 at SNR
    1366, 1245, 567 and 819.
  - Found moderate sources (SNR 26–235) are at 1.2–3.0×.
  - The crowded SNR-64 source (1.1 bins from the SNR-819 one) stays at 19×.
- **The other parameters are not.** A, sky, inclination and phases of loud sources stay at
  **3–9×** (median 3.2–6.5), and 12× for the crowded SNR-64 source. The features remove the
  frequency coordinate's precision limit, not the others'. Those are O(1) bf16 coordinates
  too: unit vectors on S², the amplitude in [−1, 1]. The same mechanism is the natural
  suspect.
- **The same six sources are missed.** Their f₀ ratios are 45–330, and 70–89% of their
  draws sit > 1 bin off.
- **Coverage is 125/128 inside the 95%** (108/128 inside the 68%), against 127/128 for the
  768 final.
  - One exception is again the missed SNR-9.3 source's sky latitude.
  - The other two are the **amplitude and longitude of the SNR-235 source in q0.30, shifted
    by 3.1 and 7.8 MCMC σ (W1)**, with widths only 1.3–2.4× the MCMC's.
  - That source sits 0.16 bins from the missed SNR-37 source: the flow folds the
    neighbour's signal into it. The wider posteriors without the features covered that
    shift; R1's sharper ones expose it. Masking now costs bias as well as detection.

**Paper.** R1 is in the following places:
- the sixth rung of the XS ladder, table and figure;
- a line in `fig:width-snr` and a row in `tab:b-coordinate`;
- `tab:mcmc-r1` with its paragraph in `app:mcmc`;
- the final numbers in `app:b` and `app:xs-crowding`;
- the abstract and the last item of `app:related`.

`style.MODELS["XS-pos"]` is red.

### F30 (2026-10-11): R1 at mid-run — the position features cut the loud floor 3.5× before any cooldown; B2 launched

**Runs.** R1 = `XS-pos` (job 14263742, `PositionedLisaFlow`, 32 periods from 2 to 128 bins),
1.2M steps from scratch with XS-late-cool's recipe. A scorecard of its epoch-474 checkpoint
(job 14275317, 4.6 min, the same 840 sources as F15–F24), at constant lr, i.e. before its
cooldown. Comparison: `outputs/scorecards/compare_r1.py`, against XS-late-cool (the same
recipe without the features, cooled, epoch 1200).

| SNR | n | found, 512-cool → R1 (768-cool) | f₀ width (bins), found by both | per-source ratio R1/512-cool [16–84%] | × ideal (R1) | in68 | in95 | rank |
|---|---|---|---|---|---|---|---|---|
| < 15 | 148 | 0.50 → 0.43 (0.54) | 0.109 → 0.090 | 0.83 [0.63–1.05] | 1.8 | 0.76 → 0.72 | 0.97 → 0.93 | 0.52 |
| 15–40 | 157 | 0.85 → 0.77 (0.86) | 0.074 → 0.049 | 0.72 [0.49–1.09] | 2.4 | 0.92 → 0.82 | 1.00 → 0.99 | 0.50 |
| 40–100 | 151 | 0.95 → 0.92 (0.97) | 0.043 → 0.019 | 0.49 [0.32–0.95] | 2.4 | 0.93 → 0.74 | 0.98 → 0.97 | 0.50 |
| ≥ 100 | 384 | 0.995 → 0.987 (0.995) | **0.039 → 0.010** | **0.28 [0.17–0.62]** | **7** | 0.93 → 0.71 | 0.99 → 0.96 | 0.48 |

- **The rule passes, half-way.** The loud floor is **0.0100 bins (95% CI 0.0093–0.0110)**,
  4.2×10⁻⁴ units of x_f, against the rule's ≤ 0.012. The per-source ratio at SNR ≥ 100 is
  0.283 (95% CI 0.256–0.317). The input representation of position was the limit: its
  encoding or its bf16 rounding. The features change both, since they are computed before the
  cast, so R1 does not separate F26's candidates 1 and 2 at the input.
- **The floor follows SNR again:** 0.0139 / 0.0103 / 0.0084 / 0.0068 bins at SNR 100–200 /
  200–400 / 400–800 / ≥ 800, against the flat 0.044–0.036 without the features. This was
  predicted.
- **Calibration is close to nominal now:** in68 0.71–0.82 and in95 0.93–0.99, where the
  baseline over-covered (in68 0.93 when loud).
- **No bias.** The loud offset is +0.0047 ± 0.0026 bins, and the window-position profile at
  most 0.004.
- **Detection lags.** R1 finds 31 fewer sources of 840 (6 found only by R1, 37 only by
  512-cool). That is XS-late's level at the end of its constant lr (F15: 0.45 / 0.81 / 0.92
  against R1's 0.43 / 0.77 / 0.92), before any cooldown.

**The loss lags too, and why.**
- Until epoch ~57 R1 tracks XS-late (13474448: same recipe, seed, aux phase): 0.5733
  against 0.5766 at 55.
- Then XS-late's flow loss drops by ~0.03 in 15 epochs, the transition of F7/F10, and R1's
  never does. It is 0.5240 against 0.4763 at epochs 373–382, and the gap was still growing.
- At initialisation (CPU, 8 samples, `outputs/scorecards/init_scale.py`) the position
  projections dominate the source tokens: x_pos has rms 0.41, against 0.06 for `x_embed`. A
  source token starts as ~98% (by variance) a random high-frequency function of its own x_f,
  which buries its other ten coordinates. On the data tokens, y_pos is 0.41 against 1.21.
- The features are computed in float64, not float32 as the code comment says:
  `src/canna/__init__.py` enables x64 globally. That is harmless (the cast to bf16 comes
  after).
- A reviewer's CPU check on 40 XS-pos windows found source and token positions consistent in
  units, origin and orientation:
  - source bins match f0·T_obs − window_start to 5×10⁻¹¹ bins;
  - token column c covers channels 2c, 2c + 1;
  - the power-weighted token centroid sits −0.02 ± 0.37 bins from the source.

  So the lag is not a position bug. It points at the initial scale. **Any next run with
  features should zero-initialise `x_pos` and `y_pos`,** so it starts as the plain network.

**B2 launched now (14275499 → 14275503, scorecard 14275504, eval 14275505).**
- The rule says B reruns with the features. The margin is clear (the CI tops at 0.011), and
  every cooldown so far has lowered the floor.
- Starting at 04:15 CEST, 8 h before R1 ends, saves 8 h on an 87 h run. The GPU budget is the
  same either way.
- B2 = `configs/B-pos.yaml` (3eb92aa): B-late's recipe with `positions: 32` (periods 2 to 8192
  bins). It uses R1's tested design, default init included, so the result maps onto R1's.
- CPU smoke test (`outputs/scorecards/b_pos_smoke.py`): 75.70M parameters, finite outputs,
  2048 distinct token positions (641 in B1).

**Predictions, written now.**
- **R1 after its cooldown** (ends ~12:30 CEST, 11 Oct):
  - the loud floor 0.004–0.008 bins (the cooldown was worth ×0.44 in C1 and C2);
  - detection back to within ~10 sources of 512-cool;
  - in68 0.68–0.80 at SNR ≥ 100.
- **B2:** two hypotheses, which its scorecard tells apart.
  - (a) The precision is still a fraction of the window: 4×10⁻⁴ of x_f is 0.6 bins on B.
    Loud widths 0.3–0.8 bins, most loud sources found.
  - (b) The features make it a precision in bins, set by the finest periods (2 bins).
    Loud widths ~0.01–0.05 bins, limited by B's 1/6 of the steps.
  - Either way B2 finds most sources at SNR ≥ 40, where B1 found none.
  - **Early read:** a B2 scorecard at epoch ~60 (43 min of GPU, ~27 h in), which F26 shows is
    cheap.

### F29 (2026-10-11): crowding — detection falls with the distance to the nearest source, most of all next to a louder one

The cut by separation that §9.5 asked for. `paper/scripts/tables.py` (`xs_separation`,
`masking`) produces `tables/xs_separation.tex` and `xs_masking.tex`. It uses the scored
sources' f₀ coordinates (`paper/data/scorecards/coord.npy`, 1 unit = 24 bins on XS).
XS-late-768-cool, 840 sources:
- **Windows are crowded.** The nearest other source is a median **5.2 bins** away: within 1
  bin for 12%, within 2 bins for 21%, within 5 bins for 49%. Over the 210 windows, the
  closest pair is a median 2.5 bins apart, and closer than 12.6 bins in every window.

| nearest source (bins) | n | found, SNR < 15 / 15–40 / ≥ 40 | width at SNR ≥ 40 (bins) | × ideal | in68 | in95 |
|---|---|---|---|---|---|---|
| 0–1 | 98 | 0.35 / 0.63 / 0.95 | 0.050 | 33 | 0.85 | 0.99 |
| 1–2 | 81 | 0.19 / 0.77 / 0.98 | 0.051 | 24 | 0.86 | 1.00 |
| 2–5 | 232 | 0.38 / 0.79 / 0.99 | 0.043 | 23 | 0.88 | 1.00 |
| 5–12.6 | 276 | 0.71 / 0.98 / 1.00 | 0.035 | 14 | 0.92 | 0.99 |
| ≥ 12.6 | 153 | 0.82 / 1.00 / 1.00 | 0.033 | 11 | 0.92 | 0.98 |

- **Masking.** Of the SNR 15–40 sources within 5 bins of another source, 63% (of 41) are
  found when it is more than 3× louder, and 86% (of 44) otherwise.
- **Loud widths** grow by about half within 2 bins of a neighbour. Coverage does not degrade.
- **Delmond et al.'s two-source separations** (12.6 and 31.6 of our bins) are where the flow
  is already at its best: every source above SNR 15 is found.
- **B is not informative:** its nearest neighbour is a median 268 bins away, and nothing is
  found anyway.
- XS-late-cool (512) shows the same pattern (the same table, `outputs/scorecards/separation.py`).

### F28 (2026-10-11): MCMC on two more windows — the flow misses sources next to louder ones, which the MCMC localises

**Setup.** The F27 protocol on eval windows 2 (q0.30: SNR 9.1, 37.2, 111, 235) and 7 (q0.80:
SNR 9.3, 9.5, 64, 819). They were chosen because each holds a faint or moderate source within
0.2 bins of a loud one. Two seeds each, 14.6–18.1 min per run on the laptop GPU
(`outputs/mcmc/run_more.sh`). Comparison: `compare_mcmc.py --windows 2 7` against
XS-late-768-cool (`outputs/mcmc/compare_windows27.log`). The paper tables now carry all four
windows (`q1.00`, `q0.60`, `q0.30`, `q0.80`).

**The MCMC is converged.** The seeds agree within 4.3% in every width, and W1 ≤ 0.06 σ except
the SNR-9.3 source's 0.22 σ. R-hat is 1.0000 and ESS ≥ 1.2×10⁵.

**Fisher.**
- In f₀ it matches the MCMC within 3% at SNR ≥ 62 (all 8 such sources of the four windows),
  and within 16% below.
- In the other parameters it is off by up to 4.5 for the SNR-819 source. That source is
  nearly face-on (cos ι = 0.995), where the linearised forecast spreads along the
  degeneracies the posterior bounds.

**Flow against MCMC, four windows, 16 sources.**
- **Coverage.** The MCMC median is inside the flow's 95% for **127 of 128** source–parameter
  pairs. The exception is the sky latitude of a missed SNR-9.3 source. It is inside the 68%
  for 112 of 128.
- **Loud f₀ ratios rise with SNR:** 6.4 (SNR 111), 11 (235), 38 (567), 44 (819), 66 (1245),
  84 (1366). That is a fixed absolute floor against the MCMC's 1/SNR. The other parameters
  are 3–8.5× too wide above SNR 500.
- **Six of the 16 sources are missed by the flow (56–89% of their draws > 1 bin off), and the
  MCMC localises all six** (f₀ within 16% of Fisher).
  - Five sit within 2.6 bins of a 6–90× louder source: 14.4 next to 1245, 27.1 next to 567,
    37.2 next to 235 (0.16 bins apart), 9.1 next to 111, and 9.3 next to 819 (0.21 bins).
  - The sixth (SNR 9.5) is 4.9 bins from the other SNR-9 source.
- **A found SNR-64 source 1.1 bins from the SNR-819 one is 6–11× too wide in every
  parameter,** with 8% of its draws off. Isolated moderate sources stay at 1–2.6×.
- **Caveat on "draws off the source"** (here, and F27's "~20% of a brighter neighbour's
  draws"). These are assignments by the evaluation's relabelling (`match_sources`, which
  weighs amplitude and sky as well as f₀), not draws the flow puts elsewhere. When a slot does
  not lock on, the matching can hand it a bright source's slot, although almost every draw has
  a slot on the bright source (`app:evaluation-matching`: 99.7% in q0.60). The paper says so
  now.

**Reading.** Detection fails where a faint source shares its bins with a loud one. The flow
places the loud one only to its floor (38–84× the MCMC width), so the mismodelled part of
the loud signal is comparable to the faint source's SNR. The MCMC has no such floor. So
removing the floor (F30) should improve detection of crowded faint sources too. That is to
be checked on R1's final model, against the same four MCMC windows, which need no new MCMC.

### F27 (2026-10-10): flow against MCMC on two XS windows — never biased, too wide when loud, misses one faint source per window

**Setup** (the protocol of `paper/appendix/mcmc.tex`).
- **Windows:** the loudest eval window (q1.00, injection 9: sources at SNR 14.4, 1245, 33 and
  1366, f = 0.10 mHz) and the crowded one of `fig:corner-median` (q0.60, injection 5: SNR
  27.1, 26.3, 62 and 567).
- **Data:** eval.py's noise realisation, so the flow (`XS-late-768-cool`, its 1024 eval
  draws) saw the same data.
- **Sampler:** `lisa_checks/mcmc_xs.py`, jexplore with temperature swaps, then differential
  evolution or stretch moves at even odds.
  - 64 walkers × 6 temperatures (1–10), all four sources jointly (32 dimensions).
  - The training prior, with the chirp mass's density by quadrature (checked against 2×10⁶
    draws: χ²/dof 0.72). The float64 likelihood.
  - The walkers start from the window's Fisher draws.
  - 100k iterations of burn-in and 300k kept, thinned by 100. Two seeds per window, 14 min
    each on the laptop GPU.
- **Comparison:** `lisa_checks/compare_mcmc.py`. Every sample set is relabelled by
  `match_sources`, and ψ, φ₀ are folded by the likelihood's exact symmetries (ψ + π, and
  ψ + π/2 with φ₀ + π; the script asserts the invariance).

**The MCMC is converged.**
- Between the two seeds, every width agrees within 2% on window 5 and within 7.5% on
  window 9, with W1 ≤ 0.1 MCMC σ.
- The exception is the SNR-14 source's sky and φ₀ (W1 0.3–0.5 σ): its sky posterior is
  broad and multimodal (MCMC 2× Fisher), and the seeds weight it a little differently.
- At SNR ≥ 567, Fisher matches the MCMC in f₀ to 2%, and in most other parameters within
  20% (the amplitude–inclination and ψ–φ₀ pairs 0.8–1.4).
- So the scorecard's "× ideal" and the corner pages' Fisher are the right yardstick where
  the floor is quoted. Below SNR ~60 Fisher can be off by 2× (A–ι near face-on, the sky of
  the SNR-14 source).

**Flow against MCMC** (width = 1.4826 MAD; "other" = A, sky, ψ, sin ι, φ₀; M_c is
prior-dominated in both, ratio 0.95–1.06).

| window, source | SNR | f₀ width, flow / MCMC | other, median [range] | flow draws > 1 bin off in f₀ |
|---|---|---|---|---|
| 9, 3 | 1366 | **84** | 5.6 [4.2–6.6] | 0.3% |
| 9, 1 | 1245 | **66** | 6.4 [3.6–8.4] | 0% |
| 5, 3 | 567 | **38** | 4.8 [4.1–5.8] | 0% |
| 5, 2 | 62 | 6.2 | 2.1 [0.5–6.8] | 18% |
| 9, 2 | 33 | 4.8 | 1.6 [1.0–2.1] | 20% |
| 5, 0 | 27.1 | **231** (missed) | 5.0 [2.8–54] | 87% |
| 5, 1 | 26.3 | 2.3 | 1.1 [1.0–1.2] | 0.3% |
| 9, 0 | 14.4 | 21 | 5.3 [2.0–9.5] | 58% |

- **Never biased.** The MCMC median lies inside the flow's central 95% for all 64
  source–parameter pairs, and inside its 68% for 57 of 64. The flow is conservative, as the
  scorecards' calibration said.
- **Loud sources are too wide.**
  - f₀ is 38–84× the MCMC width: the floor of F23 and F26 (0.037 bins against
    0.0004–0.0013).
  - Everything else is 4–8× too wide, along the posterior's own degeneracies
    (`corner_window9_s3.pdf`: the flow spreads along the A–ι line).
  - So the precision limit is not only f₀'s, although f₀ is where it costs most.
- **Faint and moderate sources are near-optimal when found.** At SNR 26–33 the flow is
  within 1.0–2.1× in every parameter but f₀ (2.3–4.8×).
- **One faint source per window is missed, and that is the flow's failure, not the
  data's.**
  - The MCMC localises the SNR-27.1 source to 0.023 bins, as Fisher does (×1.06), while the
    flow is 231× wider.
  - In ~20% of its draws, the missed source's brighter neighbour (SNR 33 or 62) is placed
    elsewhere: the flow's slots get confused when one is not locked on.
- **ψ and φ₀ of the SNR-62 source** are a near-face-on degeneracy (sin ι ≈ 0.95). Only a
  combination is constrained in both, so their width ratio (0.5) is not meaningful; W1 is
  0.3 σ.
- **Cost.** 14 min of laptop GPU per window and seed, against ~1 s per window for the flow
  on an A100.

**Consequences.**
- For the paper: `tab:mcmc` and `fig:mcmc` can be filled from `outputs/mcmc/`
  (`compare_window{9,5}.npz`, `corner_window*_s*.pdf`).
- For R1: the 4–8× in the non-f₀ parameters says the floor is broader than f₀. R1's
  position features target f₀ (and the sky through the token positions). If R1 fixes f₀
  but not the rest, the remaining limit is elsewhere.

### F26 (2026-10-10): B1 — B trains cleanly, but f₀ stops at ~3 bins: XS's floor, in coordinate units

**Training (jobs 13882121 → 13882126, 6–10 Oct).** `B-late`, 512 × 8, 200k steps, the last
40k cooling down.
- Four 24 h links did all 200 epochs. The first three hit the time limit at epochs 56, 112
  and 167, and the next link resumed within seconds each time. The spare link found the run
  complete and exited in 57 s. Training ended at 08:58 CEST on 10 Oct.
- 1.54 s per step and 18.6 GiB peak throughout, as B0 measured. No NaN.
- Restarting the optimizer at each link (no saved moments) cost nothing visible:
  0.5370 → 0.5374 at epoch 57, 0.5154 → 0.5098 at 113.
- 10-epoch mean flow loss: 0.668 (epochs 1–10), 0.538 (51–60), 0.525 (91–100), **0.498**
  (151–160, the last at constant lr), **0.473** (191–200).
  - The cooldown is worth −0.025. The constant-lr trend alone would have given about
    −0.012 over the same 40 epochs.
  - XS ended at 0.435 (512) and 0.428 (768).
- The aux losses drift once their weight reaches 0 (y ends at 1508, x at 0.25), as on every
  XS run. Sampling uses only the velocity output.

**Scorecard (job 13948963, 43 min, 10 eval + 50 random injections, 240 sources) and eval
(13949002, 41 min).** Both peaked at 5.5 GiB.

| SNR | n | found | f₀ width (bins), median | × ideal | in68 | in95 | rank |
|---|---|---|---|---|---|---|---|
| < 15 | 40 | 0.00 | 3.93 | 73 | 0.90 | 1.00 | 0.49 |
| 15–40 | 38 | 0.03 | 3.80 | 197 | 0.84 | 0.97 | 0.38 |
| 40–100 | 47 | 0.00 | 2.93 | 361 | 0.89 | 0.98 | 0.43 |
| ≥ 100 | 115 | 0.00 | 3.17 | 2148 | 0.89 | 1.00 | 0.45 |

(The widths here are over all sources with width under 50 bins, since almost none is
"found". 11 sources are lost, wider than 50 bins, all at SNR < 10.)

- **Only 1 of 240 sources is under a bin.** Widths are 2.8–4.5 bins (IQR) at every SNR, from
  5 to 1668. On XS the loud floor is 0.037 bins.
- **The posteriors are broad but honest.** in68 is 0.84–0.90 and in95 0.97–1.00 in every
  band, so they over-cover. Offsets are about 0.4 widths.
- **The offsets are shared within an injection.** The std of the per-injection mean is
  3.2 bins, against 1.6 within an injection. The loudest eval injection (q1.00) is shifted
  by +12 to +19 bins on all four sources. Below 1 mHz, widths and offsets are about 20%
  larger.
- **Loudest eval source (SNR 1668, `corner/gb-loudest.pdf`).**
  - The sky is resolved: λ to ±0.009 rad.
  - M_c is broad (±0.11), and ψ and φ₀ are flat, while the Fisher has them tight.
  - With f₀ uncertain by 3 bins the phase is lost, so that follows.
- The other per-GB pages are the median (SNR 60.8, q0.10 s3) and the faintest (SNR 7.3,
  q0.20 s1).
- The corner pages log "Too few points to create valid contours" because the Fisher draws
  are far narrower than the axes. That is cosmetic.

**What does not set the width.** Each scored source's f₀ coordinate was regenerated
(`outputs/scorecards/positions_B.py`; they match the scorecard's SNRs).
- **The bf16 cell of the input coordinate does not.** That cell is 6 bins for |x_f| > 0.5
  and shrinks toward the centre. But sources in 0.09–0.4-bin cells are as wide (2.5–3.0
  bins) as those in 6-bin cells (≈ 3.2). Spearman ρ with the cell is +0.21, and the
  widths are 30× the smallest cells.
- **The bf16 positions of the y tokens do not either.**
  - `PositionalEmbed` builds `arange(2048)` in the compute dtype. In bf16 that gives only
    **641 distinct positions for the 2048 frequency tokens**: steps of 2, 4 and 8 tokens
    above token 256, 512 and 1024. XS (32 tokens) and S (128) are exact.
  - Its highest frequency is one period per window, so on B neighbouring tokens differ by
    0.003 rad, against 0.2 on XS.
  - Yet sources sit at tokens 264–1792, and the widths are 3.1, 3.1 and 3.5 bins where
    the step is 4, 8 and 16 bins.
  - It is a defect for B regardless, and the next B run should not keep it.
- **Not eval-time rounding (bound from F21).** fp32 at eval left XS-late's width unchanged
  (0.991, 95% ≥ 0.978). That caps eval-time rounding at 7.5×10⁻⁴ coordinate units in
  quadrature, so fp32 at eval could narrow B by at most ~7%. The test is not worth running.
  It would also need smaller chunks, since fp32 attention falls back to XLA's materialised
  logits.

**What does fit: a fixed precision in the flow's coordinate.** The f₀ coordinate x_f maps
the window interior onto [−1, 1].
- One unit of x_f spans **24 bins on XS and ~1536 on B**.
- In those units:

| run | loud f₀ width (bins) | in units of x_f |
|---|---|---|
| XS-late-768-cool (F23) | 0.037 | **1.5×10⁻³** |
| XS-late-cool (F24, 512) | 0.039 | **1.6×10⁻³** |
| B-late (512, SNR ≥ 40) | 3.0 | **2.0×10⁻³** (IQR 1.7–2.4) |

- B, with a 64× wider window and a sixth of the steps, places sources to within 30% of
  XS's best precision in x_f. That precision is about half a bf16 ulp of an O(1) number
  (2⁻⁸ = 3.9×10⁻³).
- The floor F6 and F23 describe on XS, 25× Fisher, is the same limit. On XS it sits below
  a bin; B's window makes it 3 bins.
- To find sources on B (width < 1 bin) the network needs 6.5×10⁻⁴ units, 2.5× finer than
  XS's best. To reach Fisher for loud sources it needs ~10⁻⁶.
- On XS only the cooldown has moved this number (3.6 → 1.5×10⁻³). Width (F19, F24),
  integrator steps (F16) and fp32 at eval (F21) did not.

**Candidate causes, both structural.**
1. **Representation.** x_f enters `x_embed` as a raw scalar, through an MLP, with no
   Fourier features. The y positions come from sinusoids no finer than the window. Both
   resolve a fixed fraction of the window, whatever its size in bins.
2. **bf16 in training** (F5, F21, still open). After the first layer the coordinate
   lives in O(1) bf16 activations, and the gradient cannot shape structure below their
   rounding.
- Fourier features of x_f, computed in fp32 with periods down to about a bin, address
  both. They carry the fine position in O(1) features, so the bf16 cast no longer limits
  it.

**Consequences.**
- B as built cannot localise f₀. More steps or the 768 width would not close a 1000×
  gap: B already matches XS's best coordinate precision.
- §9's claim 5 (one network for the whole band) is not supported yet.
- The fix is architectural, and XS can test it cheaply: its floor is the same limit (§6).

### F25 (2026-10-06): B0 — B costs 1.57 s per step at 512 wide and 2.57 s at 768; the eval needs chunking

**Training benchmark.** Jobs 13875733 (`B-late`, 512) and 13875737 (`B-late-768`), 300 steps
each in epochs of 100, on one A100 at batch 256:

| width | first epoch (incl. compile) | steady epoch (100 steps) | per step | peak GPU memory |
|---|---|---|---|---|
| 512 | 224 s | 157 s | **1.57 s** | 18.6 GiB of 59 |
| 768 | 334 s | 257 s | **2.57 s** | 29.4 GiB of 59 |

- That is 41× (512) and 43× (768) an XS step. The XS/S fit's 1.5 s held.
- Memory is no constraint at batch 256. The 59 GiB is JAX's default cap of 75% of the card.
- **Cost of a B run:**

| steps (20% cooldown) | 512 | 768 |
|---|---|---|
| 100k | 44 h, 2 slots | 71 h, 3 slots |
| 150k | 65 h, 3 slots | 107 h, 5 slots |
| 200k | 87 h, 4 slots | 143 h, 6 slots |

- For comparison, a 1M-step XS run took 11 h (512) or 17 h (768). B can afford roughly a
  tenth to a fifth of XS's steps.

**Eval.** Job 13875840 ran out of memory: `sample_posterior` pushed all 1024 draws through
B's 2048 tokens at once, which needs a single 27.3 GiB buffer.
- Fixed by pushing the draws in chunks of 256 (`eval.CHUNK`). That changes nothing but the
  memory. A test checks it against the unchunked transport, to float32 rounding.
- The first chunk is exactly the scorecard's 256 draws.

**Scorecard.** Job 13875770 runs at ~78 s per injection at 768 (256 draws). The default 10 +
200 injections would take ~4.5 h on B, so **B scorecards use `--n_random 50`**: 60
injections, ~1.3 h, ~240 sources.

**Recommendation for the budget (the user decides).** At a fixed GPU budget, the 512 net
gets 1.64× the steps of the 768.
- On XS, at equal steps, the 768 net bought +11 of 840 sources and 11–15% sharper widths at
  SNR < 100 (F24).
- B will be far short of XS's 1M steps, though, and on XS each doubling of steps was worth
  ~20% in width plus the coarse gains (F12).
- **Default: `B-late` (512) for 200k steps, ~3.6 days in 4 chained slots.** The alternative
  at the same cost is `B-late-768` for ~120k.

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

## 6. Roadmap (2026-10-03, updated 2026-10-13)

### Update, 2026-10-13 evening (after XS-sfplus, F40)

Where things stand:
- **Best XS model:** R1 (WDM, position features, cooled).
- **XS-fd** matched R1 below SNR 100 with a simpler recipe (F33).
- **XS-sfplus** regressed. Folding six changes into one run has now hidden the cause of a
  regression, so the next step bisects, cheaply, before another full run.

0. **Fetch from TREX** (needs the user's `ssh trex`):
   - B-sfplus (scorecards x and y, eval) and XS-sfplus's x scorecard and eval;
   - the full XS checkpoint.

   Then delete the TREX copies once verified (the quota is at 9.0 of 9.2 GB).
1. **Fix the optimizer interaction.** No new run is needed for that alone.
   - Drop the 1/L1 weighting, or give it ScheduleFree+'s decay ∝ lr², which it needs.
   - Keep λ = 0.05 at lr 1e-3: equilibrium weight RMS ≈ 0.020, near the initial 0.0255.
2. **Bisect, three short XS runs in parallel** (300k steps, ~3–4 h wall at the 3-GPU limit),
   all with the fixed optimizer, each scored (x and y) at the end:
   - **a.** XS-fd's representation: unordered sources with the `Set` coupling (back as an
     option), 32 tokens, no reconstruction head. Against XS-fd's own loss history at equal
     steps (the same loss definition), it isolates the optimizer.
   - **b.** a + 16 bins per token: the token cut. This decides whether B can keep its 9× speed.
   - **c.** a + SNR order + the reconstruction head: the labelling.

   Rule: a change that costs more than 1.2× in median per-source f₀ width or more than 3%
   of sources found, against a, at equal steps, is reverted.
3. **Then one full XS run** (1.2M) and **one B run** with whatever passes. For B, if b fails,
   8 bins per token (512 tokens, 325 ms per step, ~18 h for 200k) is the fallback.
4. **Still open from before:**
   - masking of faint sources near loud ones (F28, F29);
   - the loud sources' A, sky and orientation at 3–9× the MCMC, limited after the
     embeddings (F33);
   - whether R1 stays the paper's final XS model.

### Update, 2026-10-12 night (after XS-fd, F33)

The recipe is simpler than R1's and as good below SNR 100. The open question is the loud
end. Is it the missing position features, or the missing cooldown, which schedule-free
did not replace?
1. **Read the y scorecard** (14401076) when it runs.
2. **The cheapest decisive test (proposed, not launched): a cooldown continuation of
   XS-fd,** about 200 epochs (~3 h), as C1 and C2 did for the WDM nets (F23, F24). It
   would start from y and decay the lr linearly to 0, with Muon and AdamW and without the
   schedule-free averaging, which would weight the cooled steps least. It needs the
   cooldown back in `train.py` as an option.
   - **Rule:** a loud floor at or below 0.0067 bins means FD + PLE replace the position
     features and the schedule should be warmup-stable-decay. Near 0.010 means the
     features are needed, and the next combined run puts them back (zero-initialised).
3. **B-fd** runs to ~16 Oct. Its scorecards come in x and in y. The same cooldown
   continuation would apply there (about 40 epochs, ~18 h).
4. **A, sky and orientation (3–9× the MCMC)** are limited after the embeddings. Candidates:
   the bf16 backbone (an fp32-eval scorecard of XS-fd on a CPU node, as T3' did), and the
   flow loss's insensitivity to them (F6).

### Update, 2026-10-12 (F32): the frequency-domain recipe on XS and B at once

The user set the recipe and asked for both runs together.
- **XS-fd** replaces R2. Same problem, seed, batch, length (1.2M) and clock as R1, so its
  scorecard compares one to one with F31.
- **B-fd** replaces B2.

**Rule for XS-fd, written before its result (R2's rule, plus "no worse than R1"):**
- the loud f₀ floor at or below 0.0067 bins;
- loud A, sky and orientation widths at most 2× the MCMC's on the four windows;
- coverage at least 125/128 inside the 95%;
- detection at least R1's.

**B-fd** passes if its loud f₀ floor falls under a bin, against B1's 2.8–4.5 bins, with
in95 ≥ 0.95.

**Intermediate reads.** Schedule-free needs no cooldown, so mid-run scorecards of XS-fd at
~400k and ~800k steps score a model as good as it will be at that point. The held-out
loss of the averaged weights should fall below the training loss at y.

### Update, 2026-10-11 afternoon (after R1's final, F31)

R1 settles the frequency floor: 4× ideal on XS, with honest coverage. Two limits remain,
both measured against the MCMC (F28, F31): the other coordinates of loud sources (3–9× too
wide), and faint sources masked by a loud neighbour (missed, and they now bias it). In order:
1. **B2 is the decisive run** (14275499 → 14275503, ~87 h, ends ~15 Oct).
   - **Early read:** a scorecard at epoch ~60 (~27 h in, i.e. ~07:00 CEST on 12 Oct; 43 min
     of GPU). It tells F30's hypotheses apart: loud widths 0.3–0.8 bins means the precision
     is in window units, ~0.01–0.05 means it is in bins. Then the final at ~15 Oct.
   - Submit it as `sbatch slurm/lisa-scorecard.sbatch B-pos --n_random 50`, and rename the
     output before the chained final one overwrites it.
2. **R2 on XS (~13 h), only after B2's early read, folding in everything we believe:**
   - zero-initialised position projections (F30's lag);
   - multi-scale fp32 Fourier features for every source coordinate, not only f₀: the sky and
     orientation vectors and the amplitude. These are F31's next limit, by the same
     mechanism;
   - an fp32 velocity head;
   - log f_w for the window-frequency embedding.

   Rule, written now: R2 succeeds if the loud sources' A, sky and orientation widths fall to
   ≤ 2× the MCMC's on the four windows (from 3–9×), with f₀ ≤ 0.007 bins and coverage
   ≥ 125/128.
3. **Masking** (F28, F29, F31) is not a precision problem: R1's sharper loud sources do not
   unmask their neighbours. It needs its own idea, e.g. training with more close pairs (the
   prior puts a pair within 1 bin in 22% of windows, within 2 bins in 39%), or a second pass on the
   residual. It belongs to the next paper unless cheap.
4. **Paper:** R1 is now the best XS model. Should it become "the final model" of the
   XS appendix (tables, figures, corner pages)? That is the user's call; for now it is the
   sixth rung, and the abstract leads with it.

### Update, 2026-10-11 (after R1's mid-run scorecard, F30)

**The floor is solved in principle.** R1's position features cut it 3.5× before the cooldown,
so step 3 below (B2) is running already. What comes next, in order:
1. **R1's final readout (~12:30 CEST, 11 Oct; scorecard and eval are chained).**
   - Apply the rule. Check the cooldown's effect on detection and calibration against F30's
     predictions.
   - Rerun `compare_mcmc.py` on the four MCMC windows with R1's eval draws: no new MCMC.
     F28's reading predicts fewer misses next to loud sources.
2. **B2 early read:** a scorecard at epoch ~60 (~27 h in, 43 min of GPU). It tells F30's two
   hypotheses apart (precision in units of the window, or in bins) long before the 87 h end.
3. **Only if R1's detection lag survives its cooldown:** one XS run, R2, folding in
   everything we believe in:
   - zero-initialised position projections (F30);
   - an fp32 velocity head (the output precision of an O(1) velocity under bf16);
   - log f_w for the window-frequency embedding, which now sees f_w in Hz as a nearly
     linear, 10⁻³-sized input (`app:architecture`).

   About 13 h. Not before R1's final, to respect the compute budget.
4. **No more MCMC needed for the paper.** Four windows (16 sources, 128 pairs) back every
   claim. The R1 comparison reuses them.

**Paper review (2026-10-11), all fixes applied in the paper.** A reviewer read a fresh
snapshot; every claim below was checked here before fixing.
- **Critical.**
  - The loss-sensitivity appendix had "about 100 times the ideal width at SNR 1000": it is
    ~300× (94× at the median loud source).
  - The proposition is asymptotic in σ/σ_m. Quadrature gives exact/asymptote 0.59 for p = 3
    at the real floor (σ/σ_m = 0.04), and 0.92 for p = 1. So the floor costs ~1.1×10⁻³ of
    the logged loss under the warp (twice the scatter), not 2×10⁻³ (four times). The proof's
    "O(σ)" is a relative O((σ/σ_m)^{1/p}).
- **Major.**
  - The abstract was stale after F28/F30.
  - "The warp alone" is confounded with the shorter aux phase.
  - On B the bf16 cell does add ~20% in the largest cells, as cell/√12 in quadrature
    predicts: 2.9 against 3.4 bins.
  - R1's features also remove input rounding (wording above).
  - "Mc not measured anywhere in B" rested on one source.
  - The SNR-62 source's cos ι is 0.98, not 0.95.
  - A citation was missing: FMPE's bounded power-law time prior (Dax+23 §3.3) already
    up-weights late times, and the warped clock is a stronger, divergent version.
- **Minor:** B is 40× XS per step, not 41. B had a sixth of the cooled XS steps. 1543 s per B
  epoch. The 18% matching contamination widens the MAD by ~25%. The chirp drift is resolved
  to √180/(πρ) ≈ 4.3/ρ bins. The MCMC table's "other" column wrongly included f₀ (no printed
  change), and its asterisk now sits on the failing entry.
5. **Beyond this paper:**
   - a variable number of sources (a SlotFlow-like cardinality head, or empty slots);
   - stitching windows into a global fit;
   - gaps and non-stationary noise (cf. Mao et al.);
   - LDC data.

### Update, 2026-10-10 (after B1, F26)

B trains, but places f₀ only to ~3 bins. That is the precision XS reaches in the flow's
coordinate (≈ 2×10⁻³ of a unit), scaled by B's 64× wider window. So the next step is
architectural, not more steps or width.

1. **The change (code, no GPU). Done in f55ab93** as `PositionedLisaFlow`, a subclass, so
   `LisaFlow`'s fields, and every checkpoint saved from it, stay as they were.
   - XS-late-768-cool and B-late restore and give bit-identical outputs on fixed inputs.
   - 374 lisa tests pass, 9 of them new (`tests/lisa/test_positions.py`).
   - What was built differs from the sketch below in two ways. The source position is
     taken from x_f through the exact window geometry (`WindowGeometry.source_bins`), not
     as octaves of x_f. And the shortest period is 2 bins, the same for both streams.

   The sketch, as written before the code. Behind network options whose defaults keep
   every existing checkpoint's function bit for bit:
   - **Fourier features of the f₀ coordinate,** sin and cos of π·2ᵏ·x_f for k = 0 … K, with
     K set from the window so the finest period is about a bin (K ≈ 5 on XS, 11 on B).
     They are computed in fp32 before the bf16 cast and concatenated to the raw coordinate
     in `x_embed`.
   - **Token positions in fp32 with periods down to 2 tokens,** instead of the current
     one-period-per-window sinusoids in the compute dtype. That fixes the 641-of-2048
     merge on B.
   - Use the same Fourier basis for the token positions and the source coordinate (one
     position, two encodings), so attention can match a source to its tokens.
   - Tests: the defaults reproduce the old outputs exactly, the features separate
     neighbouring tokens on B in bf16, and permutation invariance holds.
2. **R1: test it on XS (~13 h).** `XS-late-cool`'s recipe from scratch (512, 1.2M steps, the
   last 200k cooling down) with the features on, so it compares one to one with F24.
   - **Rule, written now:**
     - a loud floor ≤ 0.012 bins (≤ 5×10⁻⁴ units of x_f) means representation was the
       limit, and B reruns with the features;
     - a floor near 0.039 means it is not, which points to bf16 in training (F21's open
       item), to be tested next with an fp32 x-stream.
   - Its scorecard and eval as usual. One prediction is that the loud floor becomes
     SNR-dependent again.
3. **B2, only if R1 passes:** `B-late` with the features, 200k steps (87 h, 4 links).
   - If R1 cuts XS's floor by f, B's should fall to ~3/f bins. B needs f ≥ 3 to find
     sources.
4. **Meanwhile, CPU only, no GPU budget** (both from §9.5):
   - the MCMC comparison on 2–3 XS injections against `XS-late-768-cool`. Done on 10 Oct
     on the laptop GPU (F27);
   - the cut of the XS scorecards by separation from the nearest other source.
5. **Not worth running:** fp32 at eval on B (≤ 7% by F21's bound), B at 768, or a longer B
   run.

### Update, 2026-10-06 (the user's call): skip S, go to B

S1 is dropped; the next experiments are on B: the whole 0.1–12 mHz band, 4096-bin windows,
2048 tokens, 64× XS's.

**What is known about B.**
- It has never trained.
  - Job 10876424 (September) ran out of memory on the xla attention logits: 256 × 8
    heads × 2052² in fp32, 32 GiB. That is why the network now uses cuDNN flash attention.
  - Job 10876803 then hit the cuDNN CUDA-graph failure that `--xla_gpu_enable_command_buffer=`
    fixes. B has not run since.
- A 2-step smoke test of `B-late-768` on the laptop GPU (batch 2, 2026-10-06) compiles and
  trains, cuDNN attention included.
- The cost is unmeasured. The XS/S fit (`t = 23 ms + FLOPs / 105 TFLOP/s`) extrapolates to
  ~1.5 s per step at 512 wide, ~40× XS. That is ~4 days for 200k steps, and it is only a fit.

**Plan.**
1. **Benchmark (about 20 min per job).** `B-late` and `B-late-768` for 300 steps (3 epochs
   of 100) into `outputs/bench`. Then the scorecard (`--n_random 10`) and the eval on the 768
   benchmark checkpoint. The model is untrained, so they are run only to check that the B
   eval fits in memory and to time it.
   - Training now logs `time=` per epoch and `[memory] peak` after the first. The scorecard
     and eval log their peak memory at the end.
2. **Decision.** The steady-state seconds per step at each width give the cost of 200k steps.
   The width and the length are then set against the GPU budget the user picks.
   - Default: 768 if the budget allows, per F24.
   - `cooldown_steps` stays at 20% of `total_steps`.
   - If batch 256 does not fit in memory, fall back to batch 128 (and the same number of
     samples, i.e. 2× the steps).
3. **The B run**, chained over 24 h slots with `--dependency=afterany`, then its scorecard and
   eval. If the eval runs out of memory at 1024 draws, `sample_posterior` gets a chunked
   loop.

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
| **S1** | `configs/S-late.yaml` (512) or `S-late-768.yaml`: S with the full recipe, 1M steps, the last 200k cooling down | Does the recipe carry to the 0.1–4.2 mHz band? | ~34 h in two chained slots (768, per F24) | dropped 2026-10-06: the user goes straight to B |
| **B0** | benchmark: `B-late` and `B-late-768` for 300 steps, then the B scorecard and eval on the 768 checkpoint, all into `outputs/bench` | B's seconds per step and peak memory at each width; whether the eval fits | ~1 h in all | done (F25): 1.57 / 2.57 s per step, eval chunked, scorecard `--n_random 50` |
| **B1** | `configs/B-late(-768).yaml`: the full recipe on B, length set from B0 | Does the recipe carry to the whole band? | ~87 h at 512 (B0) | done (13882121 chain, 87 h; F26): trains cleanly, loss 0.473, but f₀ widths are ~3 bins at every SNR, so 1 of 240 sources is found; the floor is XS's, in coordinate units |
| **R1** | `configs/XS-pos.yaml`: XS with `positions: 32`, the sources' and token columns' positions in bins through one fp32 Fourier basis (`PositionedLisaFlow`, f55ab93), 512 × 8, 1.2M steps, the last 200k cooling down; the same recipe as XS-late-cool | Is the fixed coordinate precision (F26) a representation limit? Rule: loud floor ≤ 0.012 bins (≤ 1/3 of F24's 0.039, i.e. ≤ 5×10⁻⁴ units of x_f) → yes, and B reruns with it | ~13 h | prepared 2026-10-10 (f55ab93), handed to TREX as `~/canna-f55ab93.bundle` |

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

**From the laptop (since 2026-10-11).** `ssh trex cmd` is refused, but the relay accepts a
shell fed on stdin, over the user's ControlMaster:

```bash
cat <<'EOF' | timeout 60 ssh -T trex
cd ~/canna; squeue -u mentasgi; sbatch slurm/lisa-scorecard.sbatch XS-pos
exit
EOF
```

Code reaches TREX as a bundle: `git bundle create X.bundle <trex-head>..fml`, `put` it over
sftp, then `git pull --ff-only ~/X.bundle fml` in `~/canna` through the same shell.

```bash
# 2026-10-12: XS-fd and B-fd (F32), after `git fetch` of the bundle + `git reset --hard`
X1=$(sbatch --parsable slurm/lisa.sbatch XS-fd)
X2=$(sbatch --parsable --dependency=afterany:$X1 slurm/lisa.sbatch XS-fd)   # spare link
sbatch --dependency=afterok:$X2 slurm/lisa-scorecard.sbatch XS-fd
sbatch --dependency=afterok:$X2 slurm/lisa-eval.sbatch XS-fd
B1=$(sbatch --parsable slurm/lisa.sbatch B-fd)          # then B2..B5 with --dependency=afterany:$B(k-1)
sbatch --dependency=afterok:$B5 slurm/lisa-scorecard.sbatch B-fd --n_random 50
sbatch --dependency=afterok:$B5 slurm/lisa-eval.sbatch B-fd
# a mid-run XS-fd scorecard; rename its npz before the final one overwrites it
sbatch --job-name=canna-lisa-scorecard-xsfd-mid slurm/lisa-scorecard.sbatch XS-fd

# 2026-10-11: R1's mid-run scorecard, then B2 (F30)
sbatch --job-name=canna-lisa-scorecard-r1mid slurm/lisa-scorecard.sbatch XS-pos
J1=$(sbatch --parsable slurm/lisa.sbatch B-pos)          # then J2..J5 with --dependency=afterany:$J(k-1)
sbatch --dependency=afterok:$J5 slurm/lisa-scorecard.sbatch B-pos --n_random 50
sbatch --dependency=afterok:$J5 slurm/lisa-eval.sbatch B-pos
```

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
# B benchmark (2026-10-06): 300 steps at each width, then the B eval and scorecard on the
# 768 one, to time them and check their memory; remove outputs/bench afterwards
J512=$(sbatch --parsable slurm/lisa.sbatch B-late --total_steps 300 --log_interval 100 --cooldown_steps 0 --output_dir outputs/bench)
J768=$(sbatch --parsable slurm/lisa.sbatch B-late-768 --total_steps 300 --log_interval 100 --cooldown_steps 0 --output_dir outputs/bench)
sbatch --dependency=afterok:$J768 slurm/lisa-scorecard.sbatch B-late-768 --output_dir outputs/bench --n_random 10
sbatch --dependency=afterok:$J768 slurm/lisa-eval.sbatch B-late-768 --output_dir outputs/bench
# S1 (dropped 2026-10-06, kept for reference) at 768 (F24), ~34 h: a chain of two jobs, the second resuming from the first's
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
- **Corner pages (2026-10-06).** `eval.py` writes `corner/q*.pdf` (full), `q*-pooled.pdf`
  and `gb-{loudest,median,faintest}.pdf`. Before this change only the full pages existed.
  The finished XS runs were backfilled locally on 2026-10-06 by
  `outputs/lisa-XS/eval-tools/backfill_corner.py`, on the laptop GPU in bf16, with eval.py's
  keys. So the draws are the ones behind each run's existing full pages.
  - The runs: XS 500k (`lisa-XS/corner-gpu`), XS 1M (`1M/lisa-XS/corner`), XS-late,
    XS-late-768, XS-late-768-cool and XS-late-cool. XS-late-cool also got its full pages,
    since its eval was never run.
  - Each run's draws are kept in `<run>/eval_draws.npz`, as [injection, draw, source,
    param] in physical units.
  - The Fisher draws are shared by all XS runs, in
    `eval-tools/fisher_eval_injections.npz`.
  - Over the 40 eval sources, the per-GB pages are the same three GBs in every XS run:
    loudest SNR 1365.6 (q1.00, s3), median 62.4 (q0.60, s2), faintest 5.3 (q0.20, s1).
  - **Check.** On the 40 eval sources, the first 256 laptop draws reproduce each run's A100
    scorecard. The f₀ width ratio, laptop over A100, is 0.994–1.008, and the found sets are
    identical. That covers XS 1M, XS-late, -768, -768-cool and -cool; the 500k run has no
    scorecard to check against.
- **MCMC (F27).** `outputs/mcmc/` (local, ~400 MB):
  - `window{9,5}_s{0,1}.npz` with their logs: cold chains thinned by 100 (64 walkers × 3000),
    in sampler and physical coordinates, plus R-hat, τ, ESS and the acceptance per chunk;
  - `run_all.sh`, the four production runs;
  - `compare_window{9,5}.npz`: the per-source rows of `compare_mcmc.py`;
  - `corner_window*_s*.pdf`: MCMC, flow, Fisher and truth for each source.
  The scripts are tracked: `lisa_checks/mcmc_xs.py` (needs `.venv313`) and
  `lisa_checks/compare_mcmc.py` (the project venv).
- **B1 (F26).** `outputs/lisa-B-late/` holds the following, downloaded on 10 Oct:
  - `checkpoints/200` (params only, 75.6M, restored and finite on the CPU);
  - `scorecard.npz`, also copied as `outputs/scorecards/B-late.npz`;
  - `corner/` (10 full, 10 pooled and 3 per-GB pages);
  - `losses.pdf`;
  - `logs/`, with all five training links plus the scorecard and eval.

  In `outputs/scorecards/`:
  - `positions_B.py` regenerates each scored source's f₀ coordinate, bf16 cell and
    bins per unit of x_f into `coord_B.npz`, and asserts that the SNRs match the
    scorecard's;
  - `width_B.py` makes the width-against-cell, SNR and f₀ cuts and the per-injection
    offset coherence;
  - `token_B.py` does the cut by the bf16 token-position step.
- **F28–F30 (2026-10-11).**
  - `outputs/mcmc/`:
    - `run_more.sh` ran windows 2 and 7 (`window{2,7}_s{0,1}.npz`, logs);
    - `compare_windows27.log` is `compare_mcmc.py --windows 2 7`;
    - its summaries `compare_window{2,7}.npz` are imported into `paper/data/mcmc`.
  - `outputs/scorecards/`:
    - `XS-pos_ep474.npz` is R1's mid-run scorecard;
    - `compare_r1.py` holds R1's rule and tables;
    - `separation.py` with `separation_2026-10-11.txt` is the crowding cut for both XS
      models and B (the paper's version is in `paper/scripts/tables.py`);
    - `init_scale.py` measures the embedding scales at init;
    - `b_pos_smoke.py` is B2's CPU smoke test.

## 9. Related work (literature survey, 2026-10-09)

**How the survey was done.**
- **INSPIRE-HEP citation graph:** the papers citing Korsakova+24, Katz+25, Strub+24,
  Deng+25, Srinivasan+25, Alvey+23 ("crowded") and Dax+23 (FMPE).
- **INSPIRE title and full-text searches:** LISA, Taiji, TianQin and Galactic binaries,
  crossed with neural, normalizing flow, flow matching, diffusion, transformer,
  simulation-based and amortised. Also author lists: Korsakova, Alvey, Houba, Speri,
  Delmond, and the Auckland statistics group.
- **arXiv API abstract searches** for the same terms, which also catch cs and stat papers.
- **Talk listings:**
  - the AI for Gravitational Waves workshop at CERN, 5–8 May 2026;
  - Journées LISA France, IAP, 4–5 May 2026;
  - the 3IA seminar #55 in Nice, 22 May 2026.

Every new bib entry was checked against INSPIRE and the arXiv API. They sit in
`paper/references.bib` under "Neural and simulation-based inference for LISA", plus a few
in the MCMC and generative-model sections. Nothing cites them yet. The search scripts were
scratch and are not kept.

### 9.1 The closest work: amortised posteriors for Galactic binaries

| work (bib key) | what it infers | data and model | network, objective | label switching | priors, SNR | validated against |
|---|---|---|---|---|---|---|
| **Delmond, Korsakova, Oberlin, Marsat, Basset, Dobigeon** (`delmond2026neural`, Toulouse + OCA Nice, June 2026) | 1 GB (8 parameters); 2 GBs (16) as a proof of concept; fixed, known number of sources | GBGPU (FastGB), 1 yr, A and E in the frequency domain (Re, Im), 64–512 bins per channel, no embedding net, stationary noise, no confusion noise | NSF, ResNet conditioner, 1.2–3.9×10⁸ parameters; NLL on 7×10⁶ precomputed waveforms, with the other parameters drawn on the fly; about a week per run on one A100 | ordering f₀₁ ≤ f₀₂ (their first runs gave one mode per source) | uniform; f₀ in f_c ± 2 μHz; one variant spans 0.5 mHz with f_c as an input; SNR 10–100 | ptemcee MCMC (JS divergence); P-P plots on 1000 injections |
| **Mao, Lee, Edwards** (`mao2025robust`, Auckland Statistics, Dec 2025) | 1 GB-like source, 3 parameters (A, f, ḟ); the angles are fixed | fastlisaresponse, TDI A, 30 or 90 days, gaps (80% duty cycle) | FMPE (Gaussian base, OT path, residual MLP), with a 1D CNN on the time series or a 2D CNN on the WDM log-modulus (pywavelet); 10⁴–2×10⁴ training signals | — | log-uniform boxes of ±10⁻⁵ to ±10⁻³ around the truth; SNR about 40 | MAF with the same summary net; no MCMC |
| **SlotFlow: Houba, Giarda, Speri** (`houba2025slotflow`, ETH + ESA, Nov 2025) | K ≤ 10 sinusoids (A, φ, f) with **unknown K**: a classifier over K, then K slots | toy sinusoids in white noise; no LISA response | a shared conditional NSF per slot, **factorised across slots** given a global context; 8×10⁶ training samples | Hungarian matching on the per-slot NLL | uniform | Eryn RJMCMC at K = 3: amplitude and phase agree; **frequency 2–3× broader** (blamed on an encoder bottleneck) |
| **Korsakova, Babak, Katz, Karnesis, Khukhlaev, Gair** (`korsakova2024neural`) | density estimation, not amortised over data: a flow for the Galaxy's spatial distribution (the Sangria catalogue) and one flow per verification binary, fit to its MCMC posterior; proposed as priors, MCMC proposals and a compact catalogue (not run in a sampler there) | the Sangria population; MCMC posteriors of 4 verification binaries | NSF + RealNVP blocks, ResNet conditioners | per source; a joint flow across sources is named as needed but not built | — | quality of the fits (corner plots, likelihoods) |
| **Sasli, Karnesis, ..., Stergioulas** (`sasli2026learned`, Aug 2026, Eryn team) | learned birth proposals in RJMCMC: useless while the fit is assembled, optimal at equilibrium | wavelet and pulse models of unknown order (BBH, GW150914, EEG, a Sangria MBHB); no GB benchmark | adaptive NSF (pocoMC/zuko), retrained on the cold chain | — | — | ten-seed benchmark with a stopping rule |

**What Delmond et al. find, and where it touches us.**
- Single sources at 1 and 5 mHz match MCMC.
- At 10–15 mHz, f₀, ḟ and the sky are broader than MCMC, while the amplitudes match.
- In the 0.5 mHz band, f₀ is "noticeably harder".
- Two sources:
  - Their pairs are 2×10⁻⁴ mHz apart ("unfavourable", 6.3 one-year bins, 12.6 of our
    two-year bins) and 5×10⁻⁴ mHz apart ("favourable", 31.6 of our bins).
  - The favourable pair approaches MCMC. The unfavourable pair is broad and sometimes
    biased.
  - For comparison, the closest pair in one of our windows sits a median of **2.5 bins**
    apart on XS (every window closer than their unfavourable pair), 10 bins on S, and 163
    on B. That is 4 sources drawn log-uniformly over the 3R-bin interior (simulated).
- Their stated next steps: trans-dimensional inference, embedding networks, windows over
  the full band, and non-stationary noise, glitches and gaps.

**Population-level SBI without per-source fits.**
- `srinivasan2025simulation` trains a flow on a compressed strain spectrum to infer DWD
  population parameters.
- `desanti2026inferring` does the same from the reconstructed foreground (NPE).
- Both answer a different question from ours.

**Deep-learning detection and separation (no posteriors).**
- `zhao2023space`: a self-attention network that detects and extracts every source class.
- `houba2025deep`: an encoder–decoder that separates MBHBs, GBs and glitches.
- `ma2026deep`: separation of overlapping GCBs and an EMRI, recursively.
- `niu2026extracting`: a review.
- Not in the bib: Tay+26 classifies GB types with XGBoost; Cain+26 uses an autoencoder
  for anomaly detection against the confusion background.

### 9.2 Other LISA sources, and talks with no paper yet

**MBHBs:**
- `du2024advancing`: a normalizing flow, Taiji.
- `liang2024rapid`: continuous normalizing flows trained by flow matching (linear and trig
  paths), 11-D, with confusion noise.
- `martinvilchez2025efficient`: sequential neural likelihood.
- `spadaro2026accurate`: DINGO plus importance sampling, robust up to SNR ~500, with lower
  efficiency at ~1000.
- `liang2026fluxmc`: flow matching guiding PT-MCMC.
- `sun2026robust`: Taiji, glitch-robust, with contrastive learning.
- `zhang2026prelocalization`: an NSF for TianQin early warning.

**EMRIs:** `cole2026sequential`, where TMNRE shrinks the 11-D prior volume by 10⁶–10⁷.

**SGWB:** `alvey2024simulation` and `alvey2025leveraging` (TMNRE, saqqara), and
`dimitriou2024fast`.

**Talks with no paper yet** (not in the bib):
- **James Alvey (Cambridge):** "Tackling the LISA Global Fit: Scalable Simulation-Based
  Inference and the Road Ahead", CERN AI4GW, 6 May 2026. It is a survey plus the hurdles
  to scaling; no GB paper by Alvey exists. His LISA papers are on the SGWB, EMRIs and
  overlapping signals for 3G detectors.
- **Giovanni Giarda (ETH):** "SlotFlow: Amortized Trans-Dimensional Inference towards
  LISA", CERN AI4GW, 7 May 2026.
- **Malvina Bellotti with N. Korsakova (OCA):** "Low-latency detection and parameter
  estimation of MBHBs for LISA using flow matching", Journées LISA France, 5 May 2026.
- **Iuliu Cuceu (OCA; N. Christensen, A. Lamberts):** "LISA stochastic GW component
  separation with diffusion", using a Simformer-style transformer diffusion model, 3IA
  seminar, 22 May 2026.
- **Rahul Srinivasan:** "Rapid detection and inference of EMRIs in LISA: divide and
  conquer", with a transformer, CERN AI4GW, 6 May 2026.

**The Auckland group, more broadly.** Vajpeyi, Meyer, Maturana-Russel, Liu, Lee and Aimen
mostly do Bayesian nonparametric and variational noise PSD estimation for LISA, which is not
neural. The neural work is Mao, Lee and Edwards: `mao2025robust` above, a gap-imputing
autoencoder (PRD 111, 024067), and calibration of approximate credible intervals (PRD 109,
083002).

### 9.3 Ground-based overlapping signals, label switching, symmetries

- **Overlapping signals:**
  - `langendorff2023normalizing`: an NPE for two overlapping BBHs.
  - `alvey2023crowded`: a joint TMNRE of two BBHs with 30 parameters.
  - `hu2025hierarchical`: hierarchical subtraction with NDEs.
  - `zhao2025compact`: counting and separation with a transformer.
  - Papalini, thesis 2026, not in the bib: transformers plus flows for the ET.
- **Label switching** in GW MCMC: `buscicchio2019label`, alongside `stephens2000dealing`
  and `jasra2005markov`.
- **Equivariant flow matching** (`klein2023equivariant`) aligns prior and data samples
  over permutations of identical particles. It is the ML precedent for our set alignment
  (the nearest of the 24 permutations), and the paper must cite it.
- **GW NPE lineage:** `chua2020learning`, `green2020gravitational`, `green2021complete`,
  `gabbard2022bayesian`, `dax2021real`, `dax2025real`.
- **Transformers for GW NPE:** `kofler2026flexible` (dingo-t1).
- **Reviews:** `cuoco2025applications`, `zhao2025dawning`.

### 9.4 What overlaps with canna.lisa, and what is new (as of 2026-10-09)

**Not new: cite, and do not claim.**
- An amortised neural posterior for GBs with the LISA response (Delmond+26).
- Flow matching for LISA-band signals (Liang+24 on MBHBs; Mao+25 on one GB-like source;
  FluxMC).
- The whitened frequency-domain A/E input (Delmond+26) and a WDM input (Mao+25). With
  N_t = 2, our WDM image is a re-encoding of Re/Im (the `\draftnote` in
  `paper/appendix/conditioning.tex`), so it is not a contribution.
- Permutation-invariant multi-source amortised inference (SlotFlow; the overlapping-CBC
  NPEs) and the permutation alignment in flow matching (Klein+23).
- Transformers for GW posteriors (dingo-t1).
- A fixed, known number of sources: the same limitation as Delmond+26; SlotFlow does not
  have it.

**New, as far as the survey found.**
1. **A joint, non-factorised posterior of 4 overlapping GBs per window**: 32 physical
   parameters and 44 flow coordinates, with the full jaxgb/LISA response (TDI 1.5 A/E/T),
   T_obs = 2 yr.
   - The largest amortised joint GB posterior we found is Delmond's 2 sources and 16
     parameters.
   - SlotFlow's slots are conditionally independent given the data, by construction.
   - On XS our windows are more crowded than Delmond's hardest pair.
2. **Label switching handled inside the flow-matching coupling.** Each mode has 6144
   equivalent copies (24 permutations × the ψ + kπ/2 and φ₀ + π symmetries), and the
   coupling handles them on the product manifold. The alternatives are ordering (Delmond)
   or Hungarian matching on a per-slot NLL (SlotFlow).
3. **Riemannian flow matching on [−1,1]×ℝ×[−1,1]×S²×S²×S¹** for GW parameters. We found
   no GW inference paper that uses Riemannian flow matching or flows on spheres; the GW
   NPEs use Euclidean flows on normalised parameters or (cos, sin) embeddings.
4. **Survey-like priors and SNR range.**
   - Ours: f₀ log-uniform over the band (windows placed log-uniformly), SNR log-uniform
     from 7 to 1000, and the DWD chirp-mass prior.
   - Theirs: ±2 μHz boxes and SNR 10–100 (Delmond), and boxes of ±10⁻⁵ (Mao).
5. **One network for the whole 0.1–12 mHz band** (B: 4096-bin windows, 2048 tokens).
   Delmond trains one network per band; their widest variant spans 0.5 mHz. **Not yet
   supported:** B1 trains, but its f₀ widths are ~3 bins (F26), so the claim waits for B2.
6. **Simulation on the fly in JAX:** 2.6×10⁸ windows and 10⁹ sources per 10⁶ steps,
   against fixed sets of 10⁴–10⁷.
7. **The loud-source floor: a mechanism and a fix.**
   - Delmond's f₀ and sky widths at 10–15 mHz and SlotFlow's 2–3× broad frequencies look
     like the same symptom; they blame capacity or the encoder.
   - Ours:
     - The loss-sensitivity proposition (F6) shows that flow-matching training barely sees
       the floor.
     - The warped clock and the lr cooldown cut it from 0.13–0.25 to **0.037 bins** (F15,
       F23).
     - At 768 the floor is not capacity (F19, F24).
   - This applies to any amortised estimator of loud, narrow posteriors, and may be the
     paper's most general result.
8. **Engineering new to GW inference:** MMDiT with joint data and parameter tokens, Muon
   with a WSD cooldown, and bf16 training. This is not a scientific claim, but it is
   relevant for ICML.

### 9.5 Consequences for the paper and the plan

- **Related work** has five strands:
  - amortised per-source GB inference (Delmond, Mao, SlotFlow);
  - neural density estimation inside the global-fit samplers (Korsakova+24, Sasli+26);
  - population SBI (Srinivasan+25, De Santi+26);
  - neural inference for the other LISA sources;
  - overlapping signals, label switching and equivariant flow matching.
- **The MCMC comparison (appendix N) is now required.** Delmond and SlotFlow both have one.
  The draft's `\tbd`s there are the most important gap.
- **A cheap, CPU-only addition:** cut the existing XS scorecards by the separation from the
  nearest other source in the window (in bins). That gives width, found fraction and
  calibration against separation, and places our results next to Delmond's favourable and
  unfavourable pairs (31.6 and 12.6 of our bins) without a new run.
- **Limitations to state:**
  - 4 sources per window, with no model selection; an undetectable source returns a
    masked, prior-like posterior (Fig. `fig:window`).
  - Stationary Gaussian noise with no confusion foreground, and no gaps.
  - B does not yet localise f₀ (~3 bins, F26), and the loud floor on XS is still 25×
    Fisher. F26 traces both to one fixed precision of the f₀ coordinate.
- **Timing.**
  - Korsakova's group: NPE for GBs, flow matching for MBHBs, and diffusion for the
    stochastic part.
  - The SlotFlow group: a talk "towards LISA".
  - Delmond et al. list trans-dimensional inference as their next step.
  - So "the first joint multi-GB amortised posterior with the full LISA response" is a
    claim with a short shelf life.
