# Frequency RoPE: positions on the shared frequency axis (2026-10-30)

## Takeaways

Job 66326151, `canna.lisa` at src commit 8310cb3 with absolute RoPE (`rope_angles(freq)`). Windows come from `sample_window` (`visible_snr` = 7: window centres follow p(f0) · P(SNR ≥ 7 | f0)); the sources inside a window are still drawn from the catalogue prior. 256 sources per band.

- **Attention resolves a source to about one token.** The aligned kernel (cosine similarity of a source query against a data token at frequency offset Δ) has a half width of 2.4 bins on narrow and 4.4 on full. Its first zero is at 17 and 43 bins.
- **Sources in visible windows are loud, so their Fisher widths are narrow.** The median source SNR is 1.6 (narrow) and 7.2 (full); the median Fisher σ(f0) is 0.36 bins (narrow) and 0.090 (full), 5–95% 0.11–0.82 and 0.036–0.40. All 256 sources per band are narrower than the kernel's half width; 250 narrow and all 256 full are under one bin. The Gaussian approximation is weakest below SNR 1: 58 narrow and 7 full sources.
- **A source's own token wins, and ties only at the token edge.** The margin over the best other token at the token centre is 0.82 (narrow) and 0.70 (full); at the edge between two tokens it is 0 by symmetry. The kernel goes negative past its first zero (down to -0.30 narrow, -0.38 full), so it does not decay monotonically.
- **Fine f0 comes from the Fourier features, and most sources need it.** The finest f0 period is 1/64 bin (0.0156); no Fisher width is below it (minimum 0.056 narrow, 0.020 full). 198 narrow and 249 full sources have SNR above 1, with σ 0.06–0.56 and 0.02–0.54. bf16 position noise is 2.2e-3 (narrow) and 2.4e-3 (full) bins, below every width.
- **Many rope pairs are shorter than a token.** The 32 periods run `2·n_bins^(k/31)`, from 2 bins to 256 (narrow) or 1024 (full). Under one token (16 bins): 14 of 32 on narrow, 11 on full; under two tokens: 18 and 14. Only 4 (narrow) and 2 (full) have a period that divides the patch and so cannot tell data tokens apart.
- **Source positions come from x_t, not x̂; on full the path error exceeds the median Fisher width.** On the path to the target at 1-s = 1e-3, the median position error is 0.11 bins (99%: 0.35) on full and 0.028 (0.086) on narrow, against median Fisher widths of 0.36 (narrow) and 0.090 (full) bins: the path error is 8% (narrow) and 124% (full) of the median width.
- **Position arithmetic is exact.** `source_bins` matches an f64 reference to 9e-11 (narrow) and 9e-10 (full) bins; the cosines captured from a real forward pass match the absolute-frequency position functions to 6e-8 (float32 rounding).

## What the page shows

`frequency-rope.html` is self-contained and opens from file://, with narrow/full, kernel scale and width-type toggles. Cards:

- **Which position does each stream's token get?** Data token n sits at `16n + 7.5` bins; the source stream uses the position of x_t's f0. Positions are absolute frequencies in Hz, rotated by `rope_angles`.
- **Where the periods sit against the scales f0 lives on:** the rope, frequency-feature and c periods against token spacing, Doppler width, the finest f0 period and the Fisher widths.
- **What the kernel resolves and where it aliases:** the aligned kernel over the window, fine and log scale.
- **Does a source's own token win?** The own-token margin against offset from the token centre.
- **Which pairs can tell data tokens apart?** The phase advance per token for each rope pair.
- **How finely bf16 resolves a position.**
- **How narrow the posteriors are:** Fisher f0 width against SNR, marginal or conditional.
- **How far the source position is from the target along the path.**
- **What this means.** Companion: [data tokens](../data-tokens/data-tokens.html).

## Method

`scripts/frequency_rope.py` runs the `canna.lisa` code (src commit 8310cb3) on CPU with the `narrow-B` and `full-B` configs.

- **Windows:** drawn by `sample_window` (`visible_snr` = 7), not the catalogue's `sample_f`.
- **Periods and positions:** read from the flow (`rope_periods`, `freq_periods`); the (cos, sin) of a real forward pass (`encode` then `__call__`, backbone patched out, no frame) are captured and compared with the cosine of 2π·mod(f/period, 1), f the exact absolute frequency in Hz.
- **Kernel:** a query at offset 0 against a key at offset Δ, through the code's own rotation, in the bf16 path.
- **Fisher width:** `jacfwd` of `source_segments` for 64 windows × 4 sources, marginal (weak priors on log ḟ, the prior's conditional width, and on the angles) and conditional on the other parameters.
- **Doppler half width:** f0 v/c |cos β| T + ḟ T² + 8 antenna-sideband bins, with ḟ 5σ above the prior's mean at f0 (a plotting convention: src applies no ḟ cut).
- **Path error:** distance of `source_bins` of x_t from that of the target along the flow's interpolation path.

## Rebuild

From the repo root:

```
source slurm/unity.env  # once per shell
sbatch $CANNA_SBATCH_CPU reports/architecture/frequency-rope/scripts/frequency_rope.sbatch    # data + page, ~2 min
uv run --frozen python reports/architecture/frequency-rope/scripts/build_page.py               # page only, stdlib
```

Add `--smoke` to the sbatch for a small `data/rope-smoke.json`; `build_page.py rope-smoke.json` builds the page from it.

## Data

`data/rope.json` has one entry per band. Each holds the geometry and periods, the kernel curves and margin, the bf16 error and noise, the per-source Fisher widths, the `source_bins` check, and the path error against 1-s.
