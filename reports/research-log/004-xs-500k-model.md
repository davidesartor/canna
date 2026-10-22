# F4 (2026-09-30): what the 500k XS model gets right and wrong
40 sources of the 10 eval injections, 32 RK4 steps, 256 draws. Width = MAD of f₀ in bins of
1/T_obs; full table in `outputs/lisa-XS/corner-gpu/per_source_widths_32steps.txt`.

| SNR band | n | not localised (> 0.8 bin) | median f₀ width | median width / Fisher |
|---|---|---|---|---|
| < 15 | 6 | 6 | 4.0 bins | 72× |
| 15–40 | 13 | 5 (3 prior-wide at ~9 bins, 2 at ~1 bin) | 0.36 | 14× |
| 40–60 | 1 | 0 | 0.15 | 15× |
| ≥ 60 | 20 | 0 | 0.18 (0.13–0.25 for 18 of the 20; 0.34 and 0.61 at SNR 62–64) | 92× (18–410×) |

- Sky widths 1.5–56× Fisher. Degeneracies are captured.
- Slight low bias of loud-source f₀: 15 of 20 medians within 0.8 own width of the truth, 13 of
  20 low.
- The 3 prior-wide sources (SNR 17–37) all sit in crowded windows.

![f0 width vs SNR](figures/xs-500k-f0-width-vs-snr.png)
