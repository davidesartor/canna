# F31 (2026-10-11): R1 final — the floor drops to 4× ideal with near-nominal coverage, and detection recovers

- Loud f₀ floor **0.0067 bins** (4.2× ideal), 5.3× narrower than without features; coverage
  near nominal, no bias, detection +7 of 840. F30's predictions held.
- Against the MCMC, f₀ is fixed (5–12×) but the other parameters stay 3–9× too wide.
- Crowding now sets the remaining width; misses next to louder sources do not improve, so F28's
  masking reading is not supported.

**Runs.** R1 training (14263742) done: 13 h 18 min, 38 s per epoch, no NaN. Scorecard 14263743,
eval 14263745. In `outputs/lisa-XS-pos`: `checkpoints/1200` (776 MB, restores locally as
`PositionedLisaFlow`), `corner/`, `losses.pdf`, logs, and `eval_draws.npz` regenerated on the
laptop with eval.py's keys (`outputs/lisa-XS/eval-tools/eval_draws.py`; reproduces the stored
768-cool draws to a median 0.004 bins, width ratio 1.0001). Compared with
`outputs/scorecards/compare_r1.py` against XS-late-cool on the same 840 sources.

| SNR | n | found, 512-cool → R1 (768-cool) | f₀ width (bins), found by both | ratio R1/512-cool [16–84%] | × ideal (R1) | in68 | in95 |
|---|---|---|---|---|---|---|---|
| < 15 | 148 | 0.50 → **0.53** (0.54) | 0.118 → 0.090 | 0.74 [0.58–0.94] | 1.8 | 0.76 → 0.76 | 0.97 → 0.97 |
| 15–40 | 157 | 0.85 → 0.85 (0.86) | 0.077 → 0.044 | 0.56 [0.40–0.74] | 1.9 | 0.92 → 0.78 | 1.00 → 0.97 |
| 40–100 | 151 | 0.95 → 0.96 (0.97) | 0.043 → 0.016 | 0.40 [0.27–0.65] | 2.0 | 0.93 → 0.77 | 0.98 → 0.97 |
| ≥ 100 | 384 | 0.995 → 0.995 (0.995) | **0.039 → 0.0067** | **0.19 [0.10–0.46]** | **4.2** | 0.93 → 0.71 | 0.99 → 0.96 |

- **Floor:** 95% CI 0.0061–0.0076 bins, 2.8×10⁻⁴ units of x_f. Loud per-source ratio 0.189 (95%
  CI 0.170–0.206); 5.5× narrower than the 768 final in median width.
- **Follows SNR:** 0.0100 / 0.0063 / 0.0048 / 0.0037 bins at SNR 100–200 / 200–400 / 400–800 /
  ≥ 800 — ×2.7 over ~6.7× in SNR, slower than 1/SNR, so a residual floor remains at 4× ideal.
- **Coverage** in68 0.71–0.78, in95 0.96–0.97 (baseline in68 0.93). Rank 0.50–0.53.
- **No bias:** +0.0012 ± 0.0017 bins, window-position profile ≤ 0.002.
- **Detection:** net +7 vs 512-cool (17 vs 10), close to the 768 final at the 512's cost (13.3
  h vs 10.9 + 2.2).
- **Loss:** 0.4403 vs 0.4352 for 512-cool. The lag (0.048 at epoch ~380, 0.008 at 1000, 0.005
  after cooldown) closed during the constant-lr phase.

**Crowding** (F29's cut, `paper/scripts/tables.py`).
- At SNR ≥ 40 found widths are **0.0062 bins** beyond 5 bins of a neighbour, 0.017 at 2–5,
  0.028 within 2; for the baselines the floor hid this (0.036 / 0.046 / 0.057).
- SNR 15–40 sources within 5 bins of a > 3× louder one: 63% found, as for the 768 final (59%
  for 512-cool). The misses have another cause, perhaps slot assignment or a coarse search near
  a dominant peak.

**Against the MCMC** (F28's four windows; `compare_mcmc.py --flow
outputs/lisa-XS-pos/eval_draws.npz --tag _r1`, `outputs/mcmc/compare_r1.log`, `tab:mcmc-r1`).
- **f₀ fixed:** loud sources 38–84 → **5–12×** (7.3, 5.3, 5.0, 12 at SNR 1366, 1245, 567, 819);
  found moderate ones (SNR 26–235) 1.2–3.0×; the crowded SNR-64 source stays at 19×.
- **Other parameters not:** A, sky, inclination, phases of loud sources **3–9×** (median
  3.2–6.5), 12× for the SNR-64 one. They are O(1) bf16 coordinates too (unit vectors on S²,
  amplitude in [−1, 1]); the same mechanism is the natural suspect.
- **Same six sources missed:** f₀ ratios 45–330, 70–89% of draws > 1 bin off.
- **Coverage 125/128 in the 95%** (108/128 in the 68%), vs 127/128 for the 768 final.
  Exceptions: the missed SNR-9.3 source's sky latitude, and the **amplitude and longitude of
  the SNR-235 source in q0.30, shifted 3.1 and 7.8 MCMC σ (W1)** with widths only 1.3–2.4×
  MCMC. It sits 0.16 bins from the missed SNR-37 source and absorbs its signal; the wider
  baseline covered the shift, R1's sharper posterior exposes it. Masking now costs bias as well
  as detection.

**Paper.** R1 is the sixth rung of the XS ladder (table, figure), a line in `fig:width-snr`, a
row in `tab:b-coordinate`, `tab:mcmc-r1` with its `app:mcmc` paragraph, the final numbers in
`app:b` and `app:xs-crowding`, the abstract and the last item of `app:related`.
`style.MODELS["XS-pos"]` is red.
