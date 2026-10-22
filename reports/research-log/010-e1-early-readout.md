# F10 (2026-10-01): E1 early readout — no jump when the aux weight crosses 0.2
- Prediction E1(i) failed: no fast drop at E1's crossover. F7's mechanism is wrong as stated,
  and the capacity question (Q1) is open again.

E1 = job 13395354, `--warmup_frac 0.1`, otherwise the baseline (same seed). Its aux weight
crossed 0.2 at epoch 36 and reached 0 at 50.

| epoch | E1 aux | E1 flow | baseline aux | baseline flow |
|---|---|---|---|---|
| 12 | 0.885 | 0.601 | 0.995 | 0.593 |
| 24 | 0.563 | 0.564 | 0.979 | 0.563 |
| 36 | 0.206 | 0.548 | 0.952 | 0.550 |
| 45 | 0.035 | 0.541 | 0.925 | 0.545 |
| 57 | 0 | 0.534 | 0.881 | 0.541 |

- Since then E1 falls ~5×10⁻⁴ per epoch, vs the baseline's 3.6×10⁻⁴ at the same epochs and
  2.5×10⁻³ in its drop. E1 leads by 0.007 at epoch 57. The epoch-12 gap is noise: two runs of
  one config (11000056, 12486854) differed by up to 0.04 at epoch 10.
- The aux weight alone does not trigger the jump. Either it needs training time (~175k steps in
  the baseline, coinciding with the aux decay), or the long aux phase first.
- E1 was stopped at epoch ~60, so this stays open. Running on would have told them apart: a
  drop before epoch ~175 → extra flow weight speeds the transition; near 175 → training time
  sets it; none, with a 500k loss above 0.420 → the long aux phase did real work (the two-stage
  idea in `notes.tex`).
- **Q1:** F7's aux heads not degrading only shows no trade-off with them was needed; it does
  not rule out a capacity limit on the flow itself.
