# F12 (2026-10-01): E0 posterior readout — more training helps a little, and accuracy may slip
- 2× compute shrinks widths ~22% and localises 2 more moderate-SNR sources, but the loud floor
  only moves 92× → 76× Fisher: not a practical route to Fisher widths (F6's prediction for E0
  partly wrong).
- f₀ medians sit low of the truth, more so at 1M. One source collapsed below its Fisher width.

Same 10 injections, keys, 256 draws and 32 RK4 steps as F4, laptop CPU (`widths_ckpt.py`,
`samples1M/`, `compare_500k_1M.npy`).

| | 500k | 1M |
|---|---|---|
| localised (< 0.8 bin), SNR 15–40 | 8 / 13 | 10 / 13 |
| localised, SNR ≥ 40 | 21 / 21 | 21 / 21 |
| median f₀ width, SNR ≥ 60 | 0.177 bins (92× Fisher) | 0.156 bins (76× Fisher) |
| width ratio 1M / 500k, sources localised in both | — | median 0.78 (IQR 0.72–0.87); narrower in 27 of 29 |
| sky widths / Fisher, SNR ≥ 20 (λ, β) | 10.9, 10.2 | 8.1, 8.5 |
| truth within ±1 own width of the median | 90% | 72% |
| median f₀ offset, localised SNR ≥ 20 | −0.045 bins (19 of 29 low) | −0.083 bins (23 of 31 low) |

**The low f₀ bias.**
- Not physical: chirp drift over T_obs is ~10⁻⁴ bins (at most 5×10⁻³), so no f₀–ḟ trade-off can
  shift f₀ by 0.08 bins; the offset does not correlate with the chirp-mass error (r = −0.06).
- Not a pull to the window centre (15 of 31 point that way, Spearman 0.00), so not samples
  stopping short: a uniform shift to lower f₀.
- Modest significance: sources in one injection share noise and conditioning; 8 of 10
  injections have a negative median offset (p = 0.11). Needs ~100+ injections.

**One collapse.** q0.40 source 3 (SNR 32.5): 0.060 → 0.009 bins, *narrower than its Fisher
width* (0.02), 9 widths off the truth. Over-confidence is the failure mode to watch as widths
shrink, most of all for XS-late.
