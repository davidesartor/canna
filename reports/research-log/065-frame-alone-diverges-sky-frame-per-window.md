# F65 (2026-11-02): the frame alone diverges like full-B; on the fixed catalog its galactic-latitude output becomes a per-window constant, confident and offset on faint sources, and the β-NLL's δs gradient, ∝ (x − μ)²/σ, lets those few entries run the steps

**Outcome.**
- With the flow's gradient switched off (`train_flow: false`, otherwise full-B as in
  [F63](063-narrow-b-full-b-full-b-diverged.md)), the frame diverges on the same schedule. The
  training loss bottoms at epoch ~31, rises from ~45, and the gradient norm grows from ~2 to
  4500. The rerun hit a non-finite loss at step 150 362 (epoch 151). So the flow isn't needed
  for the blow-up.
- On the fixed eval catalog (64 injections × 4 sources, saved every epoch), the failure is in
  the sky, galactic b first and then l. It isn't in f₀, which has the narrowest frame
  (δs down to −3.6) and stays clean.
- The b frame stops depending on the data: within a window μ_b varies by 0.2–0.4 prior widths
  while the true b's spread by 0.4–1.5. The whole window's (μ_b, δs_b) moves together. In y at
  epoch 100, the 1.77 mHz window sits at μ_b = +1.17 with δs_b = −2.8, and all 30 entries with
  k² > 100 that epoch are its sources, mostly at SNR 1–10. The b prior doesn't depend on the
  window, so for these faint sources the right frame is the prior.
- The suspicion that 1/σ blows up for small σ holds for the δs gradient. The loss weights each entry by sg(σ), so an
  outlier's δs gradient is 2σ|k² − 1| ≈ 2(x − μ)²/σ, which grows as the frame narrows. b's share
  of the frame gradient on the catalog grows from 1/8 to 0.44–0.56 (y, epochs 100–140), with
  single entries up to 475× the median. The μ gradient grows only as 2|k|.
- The schedule-free z noise of F63 is present at the same level (z − x RMS 0.048–0.060), so
  this run doesn't show whether the per-window drift starts in the parametrisation or in the
  noisy iterate. The drift does show in y first (epoch ~50), and in x from epoch ~100.

**The runs.** Both use full-B's defaults (8 × 512 bf16
backbone, batch 1024, 240 000 steps, warmup 4800, Muon lr 10⁻², AdamW 10⁻³, decay 0.05,
schedule-free β 0.9 → 0.965) plus `train_flow: false`. The flow's NLL is still computed and
logged, but sends no gradient, so the shared encoder learns from the frame loss alone.

| run | job | stopped | why | local |
|---|---|---|---|---|
| `261031-frame-only-full-B` (7eab5e0) | 16360391 | epoch 140 | host RAM: 17.0 GB peak against 16 GB | 33 |
| `261101-frame-only-full-B` (7c2187a, `catalog_frames: true`, 48 GB) | 16368448 | epoch 151 | non-finite loss at step 150 362 (training's guard; epoch 150's checkpoint kept) | 34 |

80 s per 1000 steps, the same as full-B: the flow's forward pass still runs for its logged NLL.
The rerun peaked at 10.6 GB of host RAM, so the first run's 17 GB isn't reproducible yet.

**History.** Per-epoch medians of the training steps at y; catalog columns at x and y; the
worst single step.

| epoch | frame (rerun) | worst step | grad norm | catalog frame, x | catalog frame, y | z − x RMS | frame (first run) | run 32: frame | run 32: grad norm |
|---|---|---|---|---|---|---|---|---|---|
| 10 | 0.511 | 0.92 | 0.8 | 0.394 | 0.397 | 0.022 | 0.481 | 0.290 | 7.2 |
| 31 | 0.200 | 1.3 | 1.7 | 0.079 | 0.120 | 0.060 | 0.201 | −0.134 | 21.6 |
| 45 | 0.276 | 1.6 | 5.1 | 0.139 | 0.217 | 0.060 | 0.263 | −0.124 | 22.1 |
| 60 | 0.839 | 6.3 × 10⁴ | 51 | 0.249 | 2.06 | 0.056 | 0.683 | 0.075 | 44.9 |
| 80 | 1.65 | 951 | 144 | 0.219 | 0.445 | 0.052 | 1.33 | 1.09 | 166 |
| 100 | 2.71 | 3.6 × 10³ | 292 | 0.499 | 16.8 | 0.048 | 1.77 | (NaN at 93) | |
| 120 | 15.9 | 1.0 × 10⁷ | 1688 | 3.04 | 1.32 | 0.048 | 4.28 | | |
| 150 | 19.2 | 1.7 × 10⁷ | 4506 | 3.70 | 1.46 | 0.050 | | | |

- The rerun's catalog frame at x is best at epoch 36 (0.059); the first run's at epoch 31
  (0.071).
- The frame alone learns slower than in full-B (0.51 against 0.29 at epoch 10), so the flow's
  gradient through the shared encoder helps the frame early on.
- The gradient norm is ~10× smaller until epoch ~45 (0.8–1.7 against 7–22). Full-B's
  gradient was mostly the flow's, always clipped at 1.

**The frame on the fixed catalog, per epoch** (`catalog_frames`: μ and log σ for x and y on
the eval catalog's 64 injections, `frames/NNN.npz`; widths δs = log σ − log σ_prior). Here
"median δs | min δs | entries with k² > 100 of 256":

| weights, epoch | f₀ | ḟ | l | b | z± (worst of 4) |
|---|---|---|---|---|---|
| y, 30 | −0.74 \| −3.56 \| 0 | −0.12 \| −0.36 \| 0 | −0.35 \| −2.43 \| 0 | −0.27 \| −1.62 \| 0 | 0 |
| y, 60 | −1.00 \| −3.29 \| 0 | −0.19 \| −0.58 \| 0 | −0.83 \| −2.15 \| 1 | −0.99 \| −2.57 \| 10 | 0 |
| y, 100 | −1.12 \| −3.45 \| 0 | −0.14 \| −0.60 \| 0 | −0.67 \| −3.01 \| 6 | −0.26 \| −2.93 \| 30 | 0 |
| y, 140 | +0.25 \| −1.42 \| 0 | +0.25 \| −0.24 \| 0 | +1.79 \| −0.81 \| 0 | +2.44 \| −0.33 \| 0 | 0 |
| x, 100 | −0.74 \| −2.74 \| 0 | −0.07 \| −0.32 \| 0 | −0.37 \| −1.28 \| 0 | −0.17 \| −0.75 \| 1 | 0 |
| x, 130 | −1.22 \| −2.61 \| 1 | −0.15 \| −0.63 \| 0 | +0.15 \| −1.05 \| 3 | +0.72 \| −0.96 \| 10 | 0 |

- **Per window, not per source.** Per-window median μ_b in y at epoch 100 is −0.31, −0.81,
  +1.17, −0.30, −0.38, +0.48, −0.14, −2.12 (windows 1.34 → 3.99 mHz). The truths' medians are
  all within ±0.11. Later the offsets grow to 4–11 prior widths (y at 140, x at 130–150).
  σ_b then widens with them (δs_b up to +5), so k² falls back but the 2δs term grows: that's
  the late loss.
- **The outliers are faint and ordinary.** Of all y outliers in b over 150 epochs, 586 are
  disk sources (|x₁,b| ≤ 1) and 232 halo sources, where halo sources are 14% of the catalog.
  Typical: SNR 3.9, x₁ = −1.29, μ = +1.20, δs = −2.79, k² = 1.7 × 10³.
- **The gradient's share** (|∂/∂δs| + |∂/∂δμ| of the weighted loss, summed per coordinate):

| weights | epoch 10–45 | 60 | 80 | 100 | 120 | 140 | max / median entry |
|---|---|---|---|---|---|---|---|
| y, b | 0.12–0.14 | 0.26 | 0.19 | 0.44 | 0.49 | 0.56 | up to 475 (epoch 100) |
| y, l | 0.13–0.14 | 0.15 | 0.14 | 0.15 | 0.17 | 0.23 | |
| x, b | 0.12–0.13 | 0.14 | 0.12 | 0.23 | 0.47 | 0.25 | up to 90 |

**Reading.** σ isn't tiny on the catalog (σ_b ≈ e^{−2.8} ≈ 0.06 prior widths at worst; F63's
training-batch probe found 10⁻⁴). What breaks is the combination:
- a b frame that has stopped reading the data and drifts per window with the weights, while
  claiming a width it has no information for;
- the β-NLL's δs gradient, (x − μ)²/σ for such an entry, which grows as that width shrinks;
- a global clip at norm 1, under which a handful of such entries set each step's direction.

**Not decided (next).** Candidates, to fold into one run:
- bound the frame's δs from below, or cap each entry's k² in the loss (a Student-t or Huber
  frame NLL), so no single entry dominates;
- tie the frame to the prior where the data are uninformative: decay δ towards 0, or predict b
  and l with a floor at the prior's width for faint sources;
- the F63 optimiser change (lower Muon lr, or decay on z for the AdamW-routed frame head), and
  then this same frame-only check, which is cheap and diverges by epoch ~50.

Code: local branch `frame-only-check`, commits 7eab5e0 (`train_flow`) and 7c2187a
(`catalog_frames`). Analysis: `outputs/runs/261101-frame-only-full-B/analysis/frames_by_epoch.py`,
whose output is `frames_by_epoch.npz`. Runs local only (`outputs/` is git-ignored).
