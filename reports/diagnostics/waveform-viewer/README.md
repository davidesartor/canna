# The LisaGB waveform viewer: prior windows, sources and noise as LisaFlow sees them

## Takeaways

- **Sources share response bins, so the flow must separate them, not just find one.**
- **φ₀ is a global rotation and nothing else.** It turns the phase of every bin, and so the hue
  of every token's phase pixel, by the same angle, and leaves every asinh|z| pixel alone.
- **ι picks which phase combination the data carry.** The signal is
  A(1 + sin ι)² e^{−iφ₋} + A(1 − sin ι)² e^{−iφ₊}, with φ± = φ₀ ± 2ψ. Going from face-on to
  face-off moves the phase the signal follows from φ₋ to φ₊.
- **The source and the data share one frequency axis.** RoPE places the source token at f₀'s bin
  and each data token at its centre bin, both counted from the window start. The rotation alone
  already favours the tokens around the source.
- **A token is a slice of the observation in time.** Each token's `patch` bins are taken back to
  the time domain. The source's Doppler sidebands become a slow winding of phase down the column.
- **What the page draws is what canna computes.** The page's JavaScript port agrees with canna to
  4 × 10⁻⁹ at worst, whole four-source windows included. The build asserts the φ₀ and ι
  takeaways and stops otherwise.

## What the page shows

`waveform-viewer.html` is one self-contained page (about 1 MB, no CDN, works from `file://`),
written by the sbatch below. Every card has a question as its title. The state (band, window,
source, slider values, every picker) lives in the URL hash, and a red banner shows any script
error.

The side panel picks:

- the band (narrow-B: 128 bins, 32 response points; full-B: 4096 bins, 1024 points);
- one of 128 prior windows per band, with ‹ › and random;
- the source in that window, sorted loudest first;
- whether the window stays put or snaps to f₀.

A slider for each of the picked source's parameters [f₀, M<sub>c</sub>, SNR, RA, Dec, ψ, ι, φ₀]
edits it, and every card recomputes in the browser. A top row picks the channel, the token
patch, the source/window view and the noise (none or a seeded draw). Tiles give the window, f₀'s
bin, the SNR and the token shape.

The cards:

- **Which sources share this window, and where do they overlap?** Each source's response bins
  and the bins holding 99% of its ρ², with overlaps shaded; clicking a row picks that source.
  Below, the whitened amplitude of each source alone and of the data, against the noise level.
- **Where does the source stand out of the noise?** The whitened window (`preprocess`), real and
  imaginary parts, for the picked source alone or the whole drawn window.
- **What does each data token hold?** The `stft_tokens` for the chosen patch: asinh|z| and the
  gated phase, one column per token.
- **Does the rotary position alone favour the tokens around the source?** `source_bins` against
  each token's centre bin, and what the rotation alone does to a score.
- **What must the flow predict, and which phase do the data carry?** The 8 flow coordinates from
  `physical_to_flow`, with φ₊ and φ₋ drawn as dials weighted by |z₊| and |z₋|.
- **Does the page's port agree with canna?** The port check's worst disagreement per quantity.
- **What this means.** The takeaways above.

## Method

[`scripts/build_viewer.py`](scripts/build_viewer.py) loads the `narrow-B` and `full-B` problems
and `LisaFlow` with `config.load`. It inlines the per-band constants, jaxgb's orbit, a JavaScript
port and each band's bank of windows.

**The port.** It covers:

- jaxgb's `JaxGB.get_tdi` (FastGB slow response, FFT, TDI 1.5, AET);
- canna's window placement, `noise_psd`, `amplitude_unit`, `preprocess`, `stft_tokens`,
  `source_bins` and `physical_to_flow`.

**The bank.** The page's 128 windows per band are `prior/scripts/draw_prior.py`'s first 128 keys,
so window i is draw i of [`../prior/data/prior.json`](../prior/data/prior.json).

**The port check.** Before writing, the build runs the port under node and checks it against
canna. It uses six prior sources per band (the loudest and quietest of three draws) and four
whole bank windows. It compares:

- the window snap and start;
- the TDI bins, the PSD, `f0_window` and `amplitude_unit`;
- the whitened window and the SNR;
- the flow coordinates, `source_bins` and the tokens at patches 4 and 16;
- each whole window's sum, joint SNR and per-source SNRs.

It also runs the φ₀ and ι invariance checks. It stops without writing if anything exceeds its
tolerance.

## How to rebuild

From the repo root, on Unity (`source slurm/unity.env` first):

```bash
sbatch $CANNA_SBATCH_CPU --constraint=avx512 reports/diagnostics/waveform-viewer/scripts/viewer.sbatch
```

`viewer.sbatch` loads Unity's `nodejs` module (avx512 nodes only) and clears stray `LISA_*`
variables. `--no-check` skips the port check, and `--out page.html` writes the page elsewhere.

## Data

None: the page draws its windows from the same keys as `prior/scripts/draw_prior.py` and holds
everything it plots.
