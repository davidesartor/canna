# F64 (2026-10-31): narrow-B finished without diverging; under the fitted prior 98% of its sources are below SNR 7, so it meets narrow-S only at SNR 7–15, where it finds all 24 at 1.6× the ideal width (narrow-S: 68% found, 0.7×, overconfident)

**Outcome.**
- narrow-B (`261030-grand-cove-narrow-B`, [F63](063-narrow-b-full-b-full-b-diverged.md)'s
  sibling of the diverged full-B) trained all 650 epochs with no non-finite step. Its eval and
  scorecard ran cleanly.
- Under the fitted (f₀, ḟ, A) catalogue prior the 0.1–1.26 mHz band is almost empty of
  detectable sources. Of the scorecard's 1056 sources, 1032 are below SNR 7 and the loudest is
  SNR 11.9; of the 1000 validation injections, 3 have a median draw SNR above 15. Nothing above
  SNR 15 can be scored, so the narrow-band results of [F59](059-oneshot-first-runs-narrow-s-full-b.md)
  (narrow-S, trained on SNR 7–1000) have no successor yet.
- Where the two overlap, at SNR 7–15, narrow-B finds all 24 sources at a median 0.11 bins,
  1.6× the ideal width, with in95 = 1.00. narrow-S found 68% at 0.038 bins, 0.7× the ideal,
  in95 = 0.78: narrower but overconfident. 24 sources is a small sample, and the two runs differ
  in prior, network size, batch and commit (below).
- The detection threshold sits at SNR 5–7: 45% found there, 10% at 3–5, ~0 below 3.
- Calibration on 1000 prior injections is good for every coordinate (mean ranks within ±0.02 of
  ½). The overlap statistic is not: the truth's template matches the data better than the
  draws' (+0.20 in the loudest slot down to +0.08), so the posterior leaves information in the
  data, consistent with the 1.4–1.8× ideal widths.
- The z runaway of F63 is present but harmless here: z − x RMS 0.060, and the median gradient
  norm at y tripled from epoch 100 to 650 while x's catalog flow kept improving.

**The run.** Davide's handoff, commit a8e3477, `narrow-B` defaults: 8 blocks × 512, batch 1024,
650 000 steps, warmup 13 000, Muon lr 10⁻², AdamW 10⁻³, weight decay 0.05, schedule-free β
0.9 → 0.965, bf16 backbone, `visible_snr: 7.0`. Job 16264497 on one A100 80 GB PCIe:
13 h 47 min, 76 s per 1000 steps. The eval (draws, 8 min 45 s) and the scorecard (7 min) ran as
two `lisa-post.sbatch` jobs in parallel at epoch 650, x weights, 32 ODE steps
(`eval/650-average-32/`).

**History.** Training losses are per-epoch medians over the 1000 steps, at y. The catalog
columns are on the fixed eval catalog, at x and at y.

| epoch | frame | flow | grad norm | max step frame | catalog frame, x | catalog flow, x | catalog flow, y | z − x RMS | x's w_rms |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.994 | −0.219 | 2.0 | 1.35 | 1.028 | −0.627 | −0.627 | 0 | 0.019 |
| 10 | 0.899 | −0.667 | 6.1 | 1.19 | 0.938 | −0.645 | −0.646 | 0 | 0.079 |
| 50 | 0.831 | −0.787 | 13.0 | 1.13 | 0.856 | −0.759 | −0.730 | 0.072 | 0.036 |
| 100 | 0.823 | −0.797 | 5.8 | 1.15 | 0.848 | −0.750 | −0.729 | 0.069 | 0.027 |
| 300 | 0.817 | −0.799 | 8.0 | 1.12 | 0.838 | −0.757 | −0.762 | 0.061 | 0.023 |
| 441 | 0.813 | −0.804 | 11.2 | 1.28 | 0.838 | −0.766 | −0.770 | 0.060 | 0.021 |
| 576 | 0.814 | −0.804 | 15.0 | 1.19 | 0.832 | **−0.768** | −0.735 | 0.060 | 0.020 |
| 650 | 0.814 | −0.803 | 16.6 | 1.10 | 0.837 | −0.765 | −0.750 | 0.061 | 0.020 |

- Training losses are flat from epoch ~100. x's catalog flow gained 0.015–0.018 over epochs
  100–650, its best at epoch 576; the eval uses the last checkpoint (−0.765), 0.003 worse.
- y is noisier than x and worse on average: catalog flow at y minus at x is +0.011 ± 0.014 over
  the last 100 epochs.
- The largest single-step frame NLL stays 1.1–1.3 after epoch ~20 (15 at epoch 18, during
  warmup). No step came near full-B's 10⁵–10²¹.

**The scorecard by SNR** (all 1056 sources; found = f₀ width under 1 bin; width and × ideal are
medians over found sources, the ideal width being √3 / (π · SNR) bins):

| SNR | n | found | width [bins] | × ideal | \|offset\| / width | in68 | in95 |
|---|---|---|---|---|---|---|---|
| 0–1 | 262 | 0.00 | – | – | – | 0.75 | 0.95 |
| 1–3 | 591 | 0.02 | 0.339 | 1.0 | 1.05 | 0.74 | 0.97 |
| 3–5 | 135 | 0.10 | 0.214 | 1.8 | 0.68 | 0.82 | 0.99 |
| 5–7 | 44 | 0.45 | 0.145 | 1.6 | 0.46 | 0.82 | 1.00 |
| 7–10 | 22 | 1.00 | 0.116 | 1.7 | 0.39 | 0.77 | 1.00 |
| 10–15 | 2 | 1.00 | 0.048 | 1.0 | 1.23 | 0.00 | 1.00 |
| 0–7, together | 1032 | 0.04 | 0.188 | 1.5 | 0.54 | 0.76 | 0.97 |
| 7–15, together | 24 | 1.00 | 0.110 | 1.6 | 0.42 | 0.71 | 1.00 |

in68 and in95 are over all sources in the band, found or not. The found medians at SNR 1–3 and
3–5 rest on 9 and 13 sources.

The validation summary's "f₀ width/ideal" of 55–108 per slot is a median over all sources,
nearly all undetected, whose f₀ posterior is the prior's ~20–30 bins; it says nothing about
detected sources.

**Against narrow-S at SNR 7–15** (narrow-S: epoch 65 of 10 000 steps, F59's scorecard):

| run | n | found | width [bins] | × ideal | \|offset\| / width | in68 | in95 |
|---|---|---|---|---|---|---|---|
| narrow-S | 123 | 0.68 | 0.038 | 0.7 | 1.21 | 0.47 | 0.78 |
| narrow-B | 24 | 1.00 | 0.110 | 1.6 | 0.42 | 0.71 | 1.00 |

Not a controlled comparison:
- Prior: narrow-S's sources were drawn at SNR 7–1000 (chirp-mass prior, `snr_range`), so its
  network knew every slot held a visible source; narrow-B's catalogue prior mostly fills the
  slots with invisible ones.
- Network: narrow-S is 6 blocks × 384, narrow-B 8 × 512.
- Batch: 4096 against 1024 at the same 650 000 steps, so narrow-S saw 4× the samples.
- Commit: 3e5bdd4 against a8e3477, which adds `response_points: 32` and drops `loss_weights`
  and the late-time path mix.

**Calibration on 1000 prior injections** (`validation.html`): for f₀, ḟ, sky and z± in all four
slots the mean rank is within ±0.02 of ½ and the share in the middle half is 0.47–0.54.
- The γ test's p is below 0.01 for ḟ in slot 0, |z₊| in all slots, |z₋| in slots 0 and 3, λ in
  slot 3, Im z₊ in slot 0 and Im z₋ in slot 3. Each deviation is ≤ 0.02 in mean rank, so 1000 injections resolve
  small shape errors, not gross ones.
- **The overlap ⟨o|h_s⟩/|h_s|** is the one clear miss: mean rank +0.20, +0.16, +0.11, +0.08
  from the loudest slot down, middle-half share 0.38–0.46. The truth's template fits the data
  better than most draws, so the draws spread over parameters the data disfavour: underconfident
  in the direction the data constrain.

**What this leaves open.**
- A narrow-band model on the fitted prior can't be judged above SNR 12 without changing the
  prior's visible-source count per window, the item the
  [fitted prior](../prior/f0-fdot-amplitude/) left for later (counts in
  [source-count](../prior/source-count/)). Until then narrow-S remains the
  narrow-band reference for loud sources.
- Whether narrow-B's 1.6× ideal at SNR 7–15 is the prior (most slots empty) or the z noise of
  F63 (y's flow is 0.011 worse than x's and its gradient norm keeps rising) isn't separable from
  this run.

Outputs, local only (`outputs/` is git-ignored): `outputs/runs/261030-grand-cove-narrow-B/`
(`config.yaml`, `history.npz`, `eval/650-average-32/{draws,scorecard}.npz`, `corner.html`,
`validation.html`). The checkpoint stays on TREX.
