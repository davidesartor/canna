# The LisaGB forward model: is the simulator right?

## Takeaways

- **The simulator is right where the flow-vs-MCMC comparison needs it.**
  - **TDI.** JaxGB's TDI-1.5 X matches Babak et al.'s equal-arm X to 1.7 × 10⁻⁴ up to 3 mHz.
    The gap grows as f₀³, to a 5.3 × 10⁻³ median in 9–11 mHz and 1.1 × 10⁻² at worst. The
    powers agree to 0.30%.
  - **Units.** The sky-averaged sensitivity over Robson et al.'s curve is flat at 1/2 below
    4 mHz: 0.4954–0.4992, slope −0.0013.
  - **Noise.** The noise generator is a standard complex Gaussian at the whitening PSD.
    ⟨n|n⟩ against χ²(4096) gives KS p = 0.83; Re against N(0, 1/2) gives p = 0.43.
  - **Response truncation.** `response_points` keeps each source's SNR to 1.5 × 10⁻⁵ but not
    its spectrum's shape: the error is 8.1% (narrow-B) and 1.35% (full-B), falling as about
    R<sup>−1/2</sup>.
  - The residual has median SNR 5.0 on narrow-B and 0.84 on full-B. Training, injections and the
    MCMC reference share that response, so it bounds realism, not the comparison.

## What the page shows

[`forward-model.html`](forward-model.html) is one self-contained page (no CDN, works from
`file://`). Every card has a question as its title; the band picker lives in the URL hash, and a
red banner shows any script error.

- **Does jaxgb's TDI match the standard equal-arm X?** The relative difference and power ratio
  per source against f₀.
- **Are the sensitivity's units right?** The sky-averaged sensitivity over Robson's curve, per
  frequency.
- **What does truncating the response cost?** Per band, each source's error against its bin
  offset, and the ladder in R.
- **Is the noise the Gaussian the likelihood assumes?** The noise generator's moments,
  cross-correlations, PSD tilt and KS tests.
- **What this means.** The takeaways above, computed from the same data.

## Method

- [`scripts/forward_model.py`](scripts/forward_model.py) records three checks:
  - **TDI.** jaxgb's X against Babak et al. 2021 eq. 14, built from the same link envelopes for
    8192 full-band prior sources.
  - **Sky average.** 256 isotropic skies at 16 frequencies over Robson et al. 2019 eq. 13.
  - **Noise generator.** 3000 pure-noise windows of 1024 bins: moments, cross-correlations, PSD
    tilt, ⟨n|n⟩ against χ²(4096), and two KS tests.
- [`scripts/response_points.py`](scripts/response_points.py) draws 4096 prior sources per band.
  It compares each band's R-bin response with a 16384-point one, and repeats that on a ladder
  R = 2⁵ … 2¹³.
- [`scripts/build_forward_model.py`](scripts/build_forward_model.py) needs only numpy: it
  summarizes `data/` into the page.

## How to rebuild

From the repo root, on Unity. The records are JAX jobs, the page is not:

```bash
sbatch --partition=cpu --account=pi_mehuber_umass_edu --qos=normal \
    reports/diagnostics/forward-model/scripts/forward-model.sbatch     # forward-model.json, tdi.npz, sky.npz
sbatch --partition=cpu --account=pi_mehuber_umass_edu --qos=normal \
    reports/diagnostics/forward-model/scripts/response-points.sbatch   # response-points.{json,npz}
uv run --frozen python reports/diagnostics/forward-model/scripts/build_forward_model.py
```

`smoke` as the first argument of either sbatch cuts the counts.

## Data

| file | written by | holds |
|---|---|---|
| `data/forward-model.json` | `forward_model.py` | the TDI, sky-average and noise-generator summary numbers, with the commit, seeds and counts |
| `data/tdi.npz` | `forward_model.py` | per-source f₀, TDI relative difference and power ratio |
| `data/sky.npz` | `forward_model.py` | the sky-averaged sensitivity over Robson's curve, per frequency |
| `data/response-points.json` | `response_points.py` | the truncation summary per band (`{band}_…`), with the commit and seed |
| `data/response-points.npz` | `response_points.py` | per-source f₀, M<sub>c</sub>, bin offset, ρ, residual ρ, relative error, outside power and the ladder errors |
