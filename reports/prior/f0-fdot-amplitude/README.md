# The LisaGB prior: f₀, ḟ and A (2026-10-30)

(log₁₀ f₀, log₁₀ ḟ, log₁₀ A) is one Gaussian fitted to the double white dwarfs of a galaxy catalogue.
A window holds 4 sources from the Gaussian inside it. The catalogue's window centre follows its f₀
marginal (`sample_f`, k = 0 on the page); `LisaGB.visible_snr = k` (`sample_window`) centres it by
about p(f₀) P(SNR ≥ k | f₀) instead, and the sources stay the catalogue's. Training uses k = 7. The
page's slider shows the windows at each k.

## Takeaways

- **The draws follow the catalogue.**
  - At k = 0 the drawn means sit within 0.0009 (narrow-B) and 0.003 (full-B) stds of the catalogue Gaussian
    truncated to the band. The contours overlap.
  - 98.0% of the catalogue lies in narrow-B, below 1.26 mHz.
- **Almost every source is invisible.**
  - 0.05% of narrow-B sources and 0.84% of full-B ones reach SNR 7.
  - 99.80% and 98.47% of windows hold none, so for them the flow's best answer is the prior.
  - The conditional mean A reaches the SNR-7 amplitude from 1.99 mHz.
- **The visible sources sit at the top of the band, where the catalogue's windows rarely go.**
  - Half the full-B windows hold a visible source only from about 1.63 mHz.
  - The upper half of the band in log f₀ gets 3.08% of the catalogue's windows: hence training's
    k = 7 window prior.
- **The cut moves windows to where sources are visible; it does not make them visible.**
  - At k = 7, 80.2% of full-B windows hold a source at SNR ≥ 7 (catalogue 1.53%), but narrow-B only
    8.1% (0.20%): below 1.26 mHz few sources can reach it.
- **ḟ matters only in full-B.**
  - 0.07% of full-B sources drift by more than a bin over the observation (narrow-B 0%).
  - Every response still fits: the widest spans 112 of 128 bins (narrow-B 32 of 32).
- **What the flow is given is a unit normal, at every cut.**
  - Per source in f₀ order, z = (x − μ) / σ in the prior's frame (`chart.prior_moments`), the frame
    the network starts from; the flow transports its N(0, 1) base to it.
  - At SNR ≥ 7 (full-B) all eight z have std 0.985 to 1.01, and 1.5e-3% of z± lies beyond ±5 (a
    unit normal: 6e-5%).
  - The cut moves the median full-B source from 10^−1.88 to 10^0.65 amplitude units, and the
    prior's frame follows it.
- **The window weights are approximate, and the resampling stays healthy in full-B.**
  - Their prior mass P(SNR ≥ k) runs under the uncut draws' share, the more so at high k: 0.69% vs
    0.84% at SNR 7 and 0.14% vs 0.23% at SNR 20 (full-B); narrow-B 0.036% vs 0.049% at SNR 7, from 16
    sources. Each candidate scores one sky and orientation, alone and at the mean ḟ.
  - Full-B keeps a median ESS / M of 0.20 at k = 7 (5% quantile 0.14), about 13 of the 64 candidates.
    Narrow-B, where the cut bites hardest, keeps 0.054 (0.027): about 3.

## The prior

- A maximum-likelihood fit to `wdwd_cat_lam_0p0_alpha_1p0_zerotides_MultiCGalaxy`, truncated at the
  catalogue's cut f₀ ≥ 0.1 mHz.
- Mean (−3.742, −20.574, −23.268); std (0.390, 1.368, 0.313).
  - corr(f₀, ḟ) = 0.991, corr(f₀, A) = 0.600, corr(ḟ, A) = 0.670.
- Given log₁₀ f₀:
  - log₁₀ ḟ has slope 3.481 per dex and residual std 0.180;
  - log₁₀ A has slope 0.481 and residual std 0.250;
  - the two residuals correlate at r = 0.720.
- `physics.SOURCE_MEAN` and `SOURCE_COVARIANCE` hold the fit.

## What the page shows

[f0-fdot-amplitude.html](f0-fdot-amplitude.html), hand-written JS with a band picker and a k slider
(hash state `#band=…&k=…`). Every card below shows the draws at the slider's cut, k = 0 being the
catalogue prior. The nav links the other prior pages.

1. **The joint prior, as training draws it.**
   - A corner of the three log parameters, with each pair as a smoothed density outlined at 1σ and
     2σ.
   - Dashed: the catalogue's Gaussian truncated to the band. The visible sources are dots.
   - Below: the fit's parameters and conditionals.
2. **Which windows hold a source the data see?**
   - Top: where the window centres fall.
   - Bottom: per centre bin, the share of windows with at least n sources at SNR ≥ 7, and the share
     of single sources.
3. **What the flow is given.** z = (x − μ) / σ of each chart coordinate in the prior's frame, against
   the flow's N(0, 1) base; hover for each std and its share beyond ±5.
4. **What the cut does.**
   - The SNR, sin ι and galactic latitude of the cut draws against the uncut ones.
   - Per f₀ bin, the prior mass P(SNR ≥ k | f₀) from the window weights against the uncut draws'
     share, and per window-centre bin the sampler's effective sample size over its M candidates.
   - A table of every k: prior mass, ESS / M, the windows with a source at SNR ≥ k, the share at
     SNR ≥ 7. Clicking a row moves the slider.
5. **What this means:** the takeaways, computed from the data.

## Method

- [`draw_f0_fdot_amplitude.py`](scripts/draw_f0_fdot_amplitude.py) runs under
  [`f0-fdot-amplitude.sbatch`](scripts/f0-fdot-amplitude.sbatch). It draws 8192 windows per band
  from `jr.split(jr.key(0), 8192)`, the keys every prior subreport uses, on
  `LisaGB(**config.load("<band>-B").problem)`.
- Per source it records the parameters, `source_snrs`, z in the prior's frame and `amplitude_unit`,
  plus the prior's curves.
- The catalogue contours come from a million numpy draws of the fitted Gaussian, truncated to the band.
- The SNR-7 amplitude is 7 · `amplitude_unit`(f₀) / √⟨κ²⟩, with κ = SNR / (A / unit) averaged over
  the draws.
- The sky average C(f₀) = ⟨ρ²⟩ at A = `amplitude_unit`(f₀), for the paper's noise and chart appendices:
  - 8192 orientations, the same at each of 16 log-spaced f₀, each source alone and monochromatic in
    the window centred on it;
  - under an isotropic and the galactic sky, with C_low the mean below 1.5 mHz (narrow-B isotropic
    3.19 ± 0.02, galactic 3.35 ± 0.02) and the ratio C(f₀) / C_low paired over the same orientations.
- The cut (`simulator.sample_window`) draws M = 64 log-uniform candidate f₀ (`VISIBLE_WINDOWS`),
  each with one prior sky and orientation, alone and monochromatic at the mean ḟ in the window
  centred on it. It picks one by p(f₀) over their density times P(SNR ≥ k | f₀, them) = Φ(−cut), cut
  being where SNR = k falls in log₁₀ A given f₀, in stds, and centres the window on it. The 4 sources
  are then drawn from the catalogue inside the window, as at k = 0.
  - `window_weights` repeats the candidates on the sampler's own key and asserts it picks the same
    centres. The prior mass reweights their Φ(−cut) by p(f₀); ESS / M is the resample's.
- The cut runs at k ∈ {1, 2, 3, 5, 7, 10, 15, 20} on the first 4096 windows' keys; the weights are
  recomputed on the first 1024 of them.
- The draws ran on the `src/` of `8310cb3`, job 66325816, in 6.8 min on 8 CPUs. The data record
  `3b8e337` with `src/` dirty: the edits `8310cb3` folds in.

## How to rebuild

From the repo root:

```bash
# redraw and rebuild (CPU, about 6 min)
sbatch --partition=cpu,cpu-preempt --account=pi_mehuber_umass_edu --qos=normal reports/prior/f0-fdot-amplitude/scripts/f0-fdot-amplitude.sbatch
# rebuild the page only (login node is fine)
uv run --frozen python reports/prior/f0-fdot-amplitude/scripts/build_page.py
```

## Data

`data/` (gitignored, about 7 min to regenerate):

- `narrow.npz` and `full.npz`: per source, the parameters, SNR, z and window centre;
- `narrow-cut.npz` and `full-cut.npz`: the same at every k, stacked on a leading axis, plus the cut's
  candidates' f₀, `log_mass` and `ess` for the weight windows;
- `f0-fdot-amplitude.json`: the prior, its curves, per-band summaries and the sky average C(f₀).
