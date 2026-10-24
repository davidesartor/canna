# The LisaGB source viewer: one source and how each parameter moves its signal, 2026-10-30

## Takeaways

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
  2 × 10⁻¹¹ at worst. The build asserts the φ₀ and ι takeaways and stops otherwise.

## What the page shows

`source-viewer.html` is one self-contained page (about 450 kB, no CDN, works from `file://`),
written by the sbatch below. Every card has a question as its title. The state (band, window,
source, its parameters, every picker) lives in the URL hash, and a red banner shows any script
error. Several sources and the noise in one window are the
[window viewer](../window-viewer/README.md); each page links to the other at the same source.

The side panel picks:

- the band (narrow-B: 128 bins, 32 response points; full-B: 512 bins, 128 points);
- the start: one of 128 prior windows per band (‹ › and random) and its source, in f₀ order, or the band's defaults (prior-centre f₀, the prior's mean ḟ there, SNR 10, a fixed orientation);
- whether the window stays put or snaps to f₀.

Three parameter blocks follow, each with a coordinate toggle; the sliders follow the active
system, and moving one holds every other coordinate of all three blocks:

- **frequency:** f₀ with ḟ (over the prior's mean ± 5σ at f₀), or f₀ with the charted ḟ, (log₁₀ ḟ − mean) / std;
- **sky:** equatorial (α, δ), ecliptic (λ, β) or galactic (l, b);
- **amplitude:** SNR (a slider over 0.1–1000, the viewer's range) or A, with ι, ψ, φ₀; Re/Im z± (the flow's own coordinates); or the polar
  asinh|z±| and φ±.

A top row picks the signal (channel h̃ Re/Im, |h̃| on a log axis, whitened, STFT tokens, or the
time domain), the channel, the token patch and the source/window view. Tiles give the window,
f₀'s bin, the SNR, ḟ and its drift in bins, A in amplitude units, the charted φ₀ and the token
shape.

The cards:

- **How does this parameter move the signal?** The picked representation, with ghosts one slider
  step down and up the chosen parameter (any slider of the active systems), the window held, and
  the distance ‖Δỹ‖ each moves the whitened window. For tokens, a third raster shows the step's
  change of asinh|z|.
- **What must the flow predict, and which phase do the data carry?** The 8 flow coordinates from
  `physical_to_flow`, with φ₊ and φ₋ drawn as dials weighted by |z₊| and |z₋|.
- **Does the rotary position alone favour the tokens around the source?** `source_bins` against
  each token's centre bin, and what the rotation alone does to a score.
- **Do the φ₀ and ι takeaways hold at this source?** The build's invariance checks, recomputed
  live at the current source beside the build's own numbers.
- **Does the page's port agree with canna?** The port check's worst disagreement per quantity.
- **What this means.** The takeaways above.

## Method

[`scripts/build_source_viewer.py`](scripts/build_source_viewer.py) loads the `narrow-B` and
`full-B` problems and `LisaFlow` with `config.load`. It inlines the per-band constants, jaxgb's
orbit, a JavaScript port and each band's bank of windows.

**The port.** It covers:

- jaxgb's `JaxGB.get_tdi` (FastGB slow response, FFT, TDI 1.5, AET);
- canna's window placement, `noise_psd`, `amplitude_unit`, `preprocess`, `stft_tokens`,
  `source_bins` and `physical_to_flow`.

The page's coordinate changes (ḟ ↔ charted ḟ, the sky rotations, z± ↔ A, ι, ψ, φ₀) are
inverted in the page and feed the same port.

**The bank.** The page's 128 windows per band are
the first 128 of `jr.split(jr.key(0), 8192)`, keyed as the prior reports draw them (e.g.
[`../../prior/f0-fdot-amplitude/scripts/draw_f0_fdot_amplitude.py`](../../prior/f0-fdot-amplitude/scripts/draw_f0_fdot_amplitude.py)),
so window i is draw i of [`../../prior/f0-fdot-amplitude/`](../../prior/f0-fdot-amplitude/README.md)'s data, its sources sorted by f₀ as training sorts them.

**The port check.** Before writing, the build runs the port under node and checks it against
canna. It uses six prior sources per band (the first and last of three draws). It compares:

- the window snap and start;
- the TDI bins, the PSD, `f0_window` and `amplitude_unit`;
- the whitened window and the SNR;
- the flow coordinates, `source_bins` and the tokens at patches 4 and 16.

It also runs the φ₀ and ι invariance checks at one fixed source per band. It stops without
writing if anything exceeds its tolerance. Whole windows are the window viewer's check.

## How to rebuild

From the repo root, on Unity:

```bash
sbatch --partition=cpu,cpu-preempt --account=pi_mehuber_umass_edu --qos=normal --constraint=avx512 reports/diagnostics/source-viewer/scripts/source-viewer.sbatch
```

`source-viewer.sbatch` loads Unity's `nodejs` module (avx512 nodes only) and clears stray
`LISA_*` variables. `--no-check` skips the port check, and `--out page.html` writes the page
elsewhere.

## Data

None: the page draws its windows from the same keys as the prior reports and holds everything it
plots.
