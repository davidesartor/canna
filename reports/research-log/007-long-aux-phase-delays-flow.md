# F7 (2026-10-01): the long aux phase delays flow learning, and it is not a capacity limit
**Revised by F10:** E1 did not jump at its 0.2 crossing, so the mechanism below is wrong as
stated, and capacity is not ruled out.

Stitched XS lineage (12486854 → 13264195 → 13378657), 9-epoch means:

| epoch | aux weight | flow | x head | y head |
|---|---|---|---|---|
| 43 | 0.93 | 0.545 | 0.184 | 0.042 |
| 170 | 0.24 | 0.514 | 0.172 | 0.036 |
| 175 | 0.21 | 0.512 | 0.172 | 0.036 |
| 210 | 0.065 | 0.456 | 0.163 | 0.037 |
| 250 | 0 | 0.442 | 0.162 | 0.040 (1.0 by epoch 260) |
| 500 | 0 | 0.420 | 0.216 | 12 |

- **Drop:** the flow loss falls 2–2.7×10⁻⁴ per epoch over epochs 43–170, then 12× faster
  (2.5×10⁻³ per epoch) once the aux weight crosses ~0.2 at epoch 176: −0.056 in 35 epochs, then
  onto the F8 power law.
- **No capacity trade-off:** through the drop the x head improves (0.172 → 0.163) and y holds
  (0.036 → 0.037). y degrades only at weight *exactly* 0 (0.005 kept it at 0.038). The aux
  tasks are cheap; their *weight* hurt.
- **Hypothesis:** Muon orthogonalises the summed gradient, so the step size is fixed and loss
  weights set only its direction. While a·‖∇L_aux‖ ≳ ‖∇L_flow‖, most of each step went to the
  aux heads.
- **Consequence:** `warmup_frac: 0.5` spends the first 35% of the budget there; 0.1 moves the
  crossover to epoch ~35 of 500.
- Against `notes.tex`'s "the jump means the network is small" hypothesis.
