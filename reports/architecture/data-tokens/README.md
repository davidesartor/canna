# Data tokens: what the data stream sees (2026-10-30)

## Takeaways

Measured on the working-tree `canna.lisa` at src commit 8310cb3 (data job 66326149). Windows come from `sample_window` (`visible_snr` = 7: window centres follow p(f0) · P(SNR ≥ 7 | f0)); the sources inside a window are still drawn from the catalogue prior.

- **One window is 8 tokens on narrow and 32 on full.** The windows are 128 and 512 bins. `stft_tokens` cuts the window into patches of 16 bins. Each token is 16 bins wide (253.5 nHz) and covers 16 time samples spaced 45.7 d apart over the 2-year observation. It has 144 features: asinh|z| and the gated phase (Re, Im) for each sample and each of the A, E and T channels.
- **Guard tokens:**
  - The guard is half the response width on each side: 16 bins (1 token) on narrow, 64 bins (4 tokens) on full.
  - That leaves 6 tokens on narrow and 24 on full where f0 can be drawn.
- **A source covers 2 to 3 tokens on narrow, 2 to 5 on full.** The central 99% of a source's SNR² spans:
  - narrow: a median of 2 tokens (5–95%: 2–3), at most 3;
  - full: a median of 3 tokens (2–5), at most 7.
  - The span grows with f0 on full: a median 3 tokens (27 bins) below 1.5 mHz, 3 tokens (35 bins) at 1.5–3 mHz, 4 tokens (54 bins) above 3 mHz. Narrow sources all sit at 0.74–1.26 mHz (median 25 bins, 2 tokens); full sources at 1.0–7.2 mHz.
- **The token at f0 holds less of the power than before.** It holds a median 78% of a source's SNR² on narrow (5–95%: 51–94%) and 57% on full (35–80%). For 4% (narrow) and 29% (full) of sources it holds less than half; on narrow the loudest token always holds at least half, on full it holds less than half for 27%.
- **Sky position barely matters; f0 sets the width.** The sources' sky prior clusters them near the ecliptic, with median |cos β| = 0.989. So the Doppler band is near its maximum for almost every source.
- **The analytic formula now slightly underestimates the width.** The measured 99% width is 1.03–1.24 (narrow) and 0.96–1.20 (full) of 2 × (f0 v/c |cos β| T + ḟ T² + 4 T/yr) at 5–95%, median 1.13 and 1.07; the extremes are 0.79–2.1 and 0.87–2.9.
- **Guard tokens carry real signal:**
  - For 22% of narrow-band sources, more than 1% of SNR² falls in a guard token, up to 64%.
  - On full this happens for 10% of sources, up to 55%.

## What the page shows

`data-tokens.html` is self-contained and opens from file://. A toggle switches between the narrow and full bands. The cards are:

- **How a window becomes tokens:** the window drawn to scale, with the token grid, the guard regions and the median source span.
- **A drawn window's tokens:**
  - Six drawn windows, as data or signal only, for channel A, E or T.
  - The feature shown is asinh|z| (sequential scale) or the gated Re or Im (diverging scale).
  - Above the heatmap: the whitened magnitude per bin, with each source's central-99% span and its f0 marked.
  - The per-bin window itself is in [the window viewer](../../diagnostics/window-viewer/window-viewer.html); this card shows only the token view.
- **Tokens per source against f0:** 1024 sources, coloured by |cos β| and plotted against the analytic span. You can switch between the 99% and 90% span and the fewest tokens holding 99% or 90%.
- **Measured against analytic width:** one point per source.
- **Power concentration:** cumulative distributions of the SNR² share in the token at f0 and in the loudest token.
- **What this means.**

## Method

`scripts/data_tokens.py` runs the `canna.lisa` code (src commit 8310cb3) on CPU with the `narrow-B` and `full-B` configs.

- **Token geometry:** read from a shape-only `LisaFlow` (`patch`, `guard`).
- **Windows:** drawn by `sample_window` (`visible_snr` = 7), not the catalogue's `sample_f`.
- **Per source:** 256 windows × 4 sources = 1024 sources per band. For each source:
  - Its own response comes from `source_segments`, and its SNR² per bin is 2|h̃|²/`window_power`.
  - The central 99% (90%) interval of the cumulative SNR² gives the span in bins and in tokens.
  - "Fewest" is the smallest number of tokens that holds the share.
  - The guard share is the SNR² in bins below `guard` or at or above `window_bins − guard`.
- **Sky:** β is the ecliptic latitude, with obliquity 23.4392911°.
- **Analytic band:** 2 × (Doppler + drift + antenna sidebands). Doppler is f0 · v⊕/c · |cos β| · T, drift is ḟ · T² (per source its prior draw; on the analytic curve ḟ 5σ above the prior's mean at f0, a plotting convention: src applies no ḟ cut), and the sidebands are `ANTENNA_HARMONICS` · T/yr = 8 bins.
- **Six drawn observations:**
  - `stft_tokens` is applied to `preprocess(sample_observation)` and to `preprocess(clean_signal)`.
  - These are stored as asinh|z| and the phase angle; the page rebuilds the gated Re and Im from them.

## Rebuild

From the repo root:

```
source slurm/unity.env  # once per shell
sbatch $CANNA_SBATCH_CPU reports/architecture/data-tokens/scripts/data_tokens.sbatch           # data + page, ~2 min
uv run --frozen python reports/architecture/data-tokens/scripts/build_page.py                 # page only, stdlib
```

Add `--smoke` to the sbatch to write a small `data/tokens-smoke.json` instead; `build_page.py tokens-smoke.json` builds the page from it.

## Data

`data/tokens.json` has one entry per band (`bands.narrow`, `bands.full`). Each entry holds:

- **Geometry:** `t_obs`, `f0_range`, `window_bins`, `patch`, `tokens`, `guard`, `features`, `response_points`.
- **`curve`:** the analytic Doppler, drift and antenna terms on an f0 grid.
- **`population`:** one row per source, with:
  - f, f0, f0_bin, snr, fdot, beta, gal_l, gal_b, doppler, drift;
  - own_share, top_share, greedy99/90, guard_share;
  - lo/hi/tokens/width for 99 and 90.
- **`observations`:** for each drawn window:
  - f, start;
  - data and signal token channels (`amp`, `phase` as [channel][token][time] ×100 integers);
  - bin_data and bin_signal;
  - the sources.

Seeds are fixed (`jr.key(1)` for the population, `jr.key(2)` for the observations).
