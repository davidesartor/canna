# The LisaGB window viewer: several sources and noise in one window, 2026-10-30

## Takeaways

- **Sources share response bins, so the flow must separate them, not just find one.**
- **What the page draws is what canna computes.** The page's JavaScript port agrees with canna to
  2 × 10⁻¹¹ at worst, whole four-source windows included.

## What the page shows

`window-viewer.html` is one self-contained page (about 400 kB, no CDN, works from `file://`),
written by the sbatch below. Every card has a question as its title. The state (band, window,
source, every picker) lives in the URL hash, and a red banner shows any script error. One source
and how each of its parameters moves the signal is the
[source viewer](../source-viewer/README.md); each page links to the other at the same source.

The side panel picks:

- the band (narrow-B: 128 bins, 32 response points; full-B: 512 bins, 128 points);
- one of 128 prior windows per band, with ‹ › and random;
- the source in that window, in f₀ order.

A top row picks the channel, the token patch, the source/window view and the noise (off, or a
seeded draw with "New draw"). Tiles give the window, each source's SNR, the window's SNR against
√Σρ², the shared bins, the noise draw and the token shape.

The cards:

- **Which sources share this window, and where do they overlap?** Each source's response bins
  and the bins holding 99% of its ρ², with overlaps shaded; clicking a row picks that source.
  Below, the whitened amplitude of each source alone and of the data, against the noise level.
- **How do the sources and the noise add up to the data?** One channel of the whitened window
  (`preprocess`) as strips: each source alone, their sum, the noise draw and the data.
- **How loud is each source, alone and in the data?** Each source's `source_snrs`, the window's
  `snr` against √Σρ², and each source's matched-filter SNR on the data.
- **What do the tokens of the whole window hold?** The `stft_tokens` of the data for the chosen
  patch: asinh|z| and the gated phase, one column per token.
- **Does the page's port agree with canna?** The port check's worst disagreement per quantity.
- **What this means.** The takeaways above.

## Method

[`scripts/build_window_viewer.py`](scripts/build_window_viewer.py) loads the `narrow-B` and
`full-B` problems with `config.load`. It inlines the per-band constants, jaxgb's orbit, a
JavaScript port and each band's bank of windows.

**The port.** It covers:

- jaxgb's `JaxGB.get_tdi` (FastGB slow response, FFT, TDI 1.5, AET);
- canna's window placement, `noise_psd`, `preprocess`, `snr`, `source_snrs` and `stft_tokens`.

**The bank.** The page's 128 windows per band are
the first 128 of `jr.split(jr.key(0), 8192)`, keyed as the prior reports draw them (e.g.
[`../../prior/f0-fdot-amplitude/scripts/draw_f0_fdot_amplitude.py`](../../prior/f0-fdot-amplitude/scripts/draw_f0_fdot_amplitude.py)),
so window i is draw i of [`../../prior/f0-fdot-amplitude/`](../../prior/f0-fdot-amplitude/README.md)'s data, its sources sorted by f₀ as training sorts them.

**The port check.** Before writing, the build runs the port under node and checks it against
canna on four whole bank windows per band. It compares each window's start, its sum of the four
whitened sources, its joint SNR, its per-source SNRs and its tokens at patches 4 and 16. It stops
without writing if anything exceeds its tolerance. The single-source quantities and the φ₀ and ι
checks are the source viewer's.

## How to rebuild

From the repo root, on Unity:

```bash
sbatch --partition=cpu,cpu-preempt --account=pi_mehuber_umass_edu --qos=normal --constraint=avx512 reports/diagnostics/window-viewer/scripts/window-viewer.sbatch
```

`window-viewer.sbatch` loads Unity's `nodejs` module (avx512 nodes only) and clears stray
`LISA_*` variables. `--no-check` skips the port check, and `--out page.html` writes the page
elsewhere.

## Data

None: the page draws its windows from the same keys as the prior reports and holds everything it
plots.
