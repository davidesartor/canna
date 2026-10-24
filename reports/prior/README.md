# The LisaGB prior: what each window asks the flow to cover (2026-10-24)

Three pages, one per parameter group: [frequency](frequency.html), [amplitude](amplitude.html)
and [sky](sky.html).

## Takeaways

### Frequency

- **Sources drift in the full band only.**
  - 99% drift less than 49 bins over the observation in the full band, and less than 0.015 in
    the narrow one.
  - M<sub>c</sub> enters the data only through that drift.
  - 22% of full-band sources drift more than one bin, and none in the narrow band.
- **The full band's f₀ marginal is reshuffled at its bottom edge.**
  - Bins 1–4 hold 0.66, 0.88, 1.13 and 1.35 of their log-uniform share.
  - The cause is window snapping: a 4096-bin window is 65 µHz, or 0.22 dex at 0.1 mHz.
- **The chirp-mass draws match the density the MCMC reference uses** (χ²/dof = 0.86 over 60 bins
  of ln M<sub>c</sub>).
- **The flow coordinates are whitened and decorrelated.** The stds are 0.576 for f₀ (uniform)
  and 0.994 for M<sub>c</sub>, with every |r| < 0.013.

### Amplitude

- **Every window is crowded and loud.**
  - Each source's SNR is log-uniform over [7, 1000], exactly: all 32768 sources drawn per band
    lie inside `snr_range`, with median 82.4 against √7000 = 83.7.
  - Windows are sorted loudest first, so slot k is an order statistic. The slot medians are
    about 450, 143, 46 and 15.
  - The window's joint SNR has median 516; only 1% of windows fall below 44.
- **Inclination alone sets κ, the SNR at one `amplitude_unit`.**
  - κ runs 1.1–2.5 (16–84%).
  - log κ correlates with |sin ι| at r = 0.99 (narrow) and 0.93 (full).
  - The sky, ψ and M<sub>c</sub> barely move it (|r| < 0.03).
  - A spans 4.3 decades in the narrow band and 5.2 in the full band (1–99%). Of that:
    - 2.15 decades is the SNR range;
    - 0.46 (narrow) to 0.61 (full) is κ;
    - the rest is the sensitivity curve.
- **The sky-averaged SNR constant is C = 3.19 ± 0.02.**
  - C rises 0.25% at the top of the narrow band (1.26 mHz) and 16% at 12 mHz.
  - The galactic sky is 5.2% louder (C = 3.35).
  - Over the training prior, ⟨κ²⟩ is 3.39 (narrow) and 3.50 (full).
- **The flow coordinates are whitened, except for z±.** Re and Im of z± are about 2.5× wider
  (std 2.46–2.49).

### Sky

- **The galactic sky prior traces the plane, not a point.**
  - 95.2% of draws have |b| < 10°.
  - Only 37%, 53%, 63% and 78% lie within 0.1, 0.2, 0.3 and 0.5 rad of the centre.
  - A quiet source's sky posterior is therefore mostly prior.
- **Other frames see the plane tilted.** 52.1% of draws have |β| < 10° (ecliptic) and 8.9%
  |Dec| < 10° (equatorial, jaxgb's frame). The draws match the quadrature density in all three
  frames (χ² per bin 0.85–1.10).
- **l and b reach the flow whitened and decorrelated** (std 1.01 / 0.99, every |r| < 0.013).
  0.88% (l) and 1.07% (b) of draws lie beyond the network's reach of ±4.

## What the pages show

Each page is self-contained (no CDN, works from `file://`) and links the other two under its
title. A band picker (narrow-B, full-B) drives every card; the band and each card's view live in
the URL hash (also `theme=light|dark`) and carry across the links. A red banner shows any script
error. Histograms carry the closed-form prior where there is one.

### Frequency

[`frequency.html`](frequency.html), about 57 KiB, e.g. `#band=full&view=drift`.

**Tiles**: the f₀ range, windows and sources per run, the share drifting beyond one bin, and the
M<sub>c</sub> check's χ²/dof.

1. **What does the prior put on f₀, M<sub>c</sub> and the drift?** One view each: f₀ against
   its log-uniform density, M<sub>c</sub> against the MCMC reference's, and the drift ḟT² in
   bins with the one-bin line.
2. **Where do the bands sit in the noise?** The noise PSD (A/E and T) with both bands' f₀ ranges.
3. **How much does one training run draw?** Windows and sources per run, both bands.
4. **What do f₀ and M<sub>c</sub> look like in the flow's coordinates?** Against the network's
   reach.
5. **What this means**: the takeaways, short.

### Amplitude

[`amplitude.html`](amplitude.html), about 192 KiB, e.g. `#band=narrow&amp=kappa&angle=psi`.

**Tiles**: the source and window SNR medians, κ (16–84%) and its correlation with |sin ι|, C at
low f₀, and ⟨κ²⟩.

1. **How loud is each source, and each window?** The per-source SNR, each slot's order
   statistic and the window's joint SNR.
2. **What amplitude does that SNR need?** The A histogram, A against f₀ with the SNR-7 and
   SNR-1000 lines at the sky average, and κ against |sin ι|.
3. **How are the orientation angles drawn?** sin ι, ψ and φ₀ against their densities.
4. **How loud is a source at one `amplitude_unit`, across f₀?** C(f₀) under an isotropic and the
   galactic sky.
5. **What do z± look like in the flow's coordinates?** Re and Im of z+ and z− against the reach.
6. **What this means**.

### Sky

[`sky.html`](sky.html), about 514 KiB, e.g. `#band=full&frame=equatorial`. A frame toggle
(galactic, ecliptic, equatorial) drives the first two cards. The bands share the sky draws.

**Tiles**: the share within 10° of the plane in each frame, within 0.5 rad of the galactic
centre, and l / b beyond the network's reach.

1. **Where on the sky does the prior put sources?** The density per steradian in the picked
   frame, Mollweide on a log colour ramp, with the galactic centre marked (and, in equatorial,
   the retired Gaussian proxy: the centre's ecliptic coordinates read as (RA, Dec)).
2. **Do the draws match the density in every frame?** Latitude and longitude marginals of the
   drawn sources against the quadrature, with χ² per bin.
3. **How far from the galactic centre do sources sit?** The cumulative share within an angle.
4. **What do l and b look like in the flow's coordinates?** Against the reach.
5. **What this means**.

## Method

[`scripts/draw_prior.py`](scripts/draw_prior.py) draws 8192 windows per band with `sample_f` and
`sample_physical` (shared key, so both bands share the SNR, mass, sky and orientation draws). It
records:

- SNRs, κ, A and the flow coordinates;
- the sky in three frames against the quadrature density;
- the chirp-mass check;
- one run's dataset size;
- C(f₀) for 8192 orientations reused at 16 frequencies, under an isotropic and the galactic sky;
- the noise PSD on a grid over both bands.

One builder per page ([`build_frequency.py`](scripts/build_frequency.py),
[`build_amplitude.py`](scripts/build_amplitude.py), [`build_sky.py`](scripts/build_sky.py)) needs
only the standard library: each inlines its part of `data/prior.json` and prints every number its
takeaways quote. The amplitude page keeps a 1500-source scatter per band; the sky page inlines
the three frames' 480×240 density maps.

## How to rebuild

From the repo root, on Unity (`source slurm/unity.env` first). The record is a JAX job, the pages
are not:

```bash
sbatch $CANNA_SBATCH_CPU reports/prior/scripts/prior.sbatch   # data/prior.json, ~6 min
uv run --frozen python reports/prior/scripts/build_frequency.py
uv run --frozen python reports/prior/scripts/build_amplitude.py
uv run --frozen python reports/prior/scripts/build_sky.py
```

## Data

| file | written by | holds |
|---|---|---|
| `data/prior.json` | `draw_prior.py` (job 66013766) | per band: SNR, κ, A, f₀, M<sub>c</sub>, ḟ, orientation and flow-coordinate histograms and quantiles, a 3000-source scatter, correlations, `sky_average` (C), `C_prior`, `chirp_mass_check`, `dataset`, `noise`; once: the sky in three frames |

The paper reads it in `paper/scripts/fig_priors.py`, `fig_chart.py`, `fig_noise.py`,
`numbers_priors.py`, `numbers_chart.py`, `numbers_noise.py` and `numbers_mcmc.py`.
