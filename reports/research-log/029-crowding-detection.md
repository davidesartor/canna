# F29 (2026-10-11): crowding — detection falls with the distance to the nearest source, most of all next to a louder one

- Windows are crowded: nearest other source a median **5.2 bins** away (within 1 bin for 12%, 2
  for 21%, 5 for 49%); the closest pair per window a median 2.5 bins, always < 12.6.
- Detection and loud width degrade near a neighbour; coverage does not.

The separation cut in [the paper section](000-takeaways.md#the-paper) asked for:
`paper/scripts/tables.py` (`xs_separation`, `masking`) → `tables/xs_separation.tex`,
`xs_masking.tex`, from `paper/data/scorecards/coord.npy` (1 unit = 24 bins on XS).
XS-late-768-cool, 840 sources:

| nearest source (bins) | n | found, SNR < 15 / 15–40 / ≥ 40 | width at SNR ≥ 40 (bins) | × ideal | in68 | in95 |
|---|---|---|---|---|---|---|
| 0–1 | 98 | 0.35 / 0.63 / 0.95 | 0.050 | 33 | 0.85 | 0.99 |
| 1–2 | 81 | 0.19 / 0.77 / 0.98 | 0.051 | 24 | 0.86 | 1.00 |
| 2–5 | 232 | 0.38 / 0.79 / 0.99 | 0.043 | 23 | 0.88 | 1.00 |
| 5–12.6 | 276 | 0.71 / 0.98 / 1.00 | 0.035 | 14 | 0.92 | 0.99 |
| ≥ 12.6 | 153 | 0.82 / 1.00 / 1.00 | 0.033 | 11 | 0.92 | 0.98 |

- **Masking:** SNR 15–40 sources within 5 bins of another are found 63% of the time (of 41)
  when it is > 3× louder, 86% (of 44) otherwise.
- **Loud widths** grow by about half within 2 bins of a neighbour.
- **Delmond et al.'s separations** (12.6 and 31.6 of our bins) are where the flow is at its
  best: every source above SNR 15 found.
- **B is not informative:** nearest neighbour a median 268 bins away, and nothing is found.
- XS-late-cool (512) shows the same pattern (`outputs/scorecards/separation.py`).
