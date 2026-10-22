# F33 (2026-10-12): XS-fd — R1's level below SNR 100, a 1.5× wider loud floor, and a lagging schedule-free average

- Below SNR 100, XS-fd matches R1. The loud floor is 1.54× R1's, so the scorecard part of its
  rule fails (floor ≤ 0.0067, detection ≥ R1).
- A, sky and orientation stay 3–9× the MCMC with fp32 inputs: F32's bf16-data hypothesis is
  refuted.
- The schedule-free average x trails the training weights y by 0.012 in flow loss.

**Run.** XS-fd (14326113, F32's recipe): 1.2M steps, 17 h 52 min, no NaN. The gradient norm
stayed at 0.2–0.6, so the clip at 1 never bound. Scorecard 14326115 and eval 14326116 score x.
Eval draws of x and y, the MCMC comparison (windows 9, 5, 2, 7) and the common-sample flow loss
were done locally.

**Scorecard against R1 (same 840 sources).**

| SNR | n | found, 512-cool / R1 / fd | f₀ width (bins), R1 → fd | per-source fd/R1 [16–84%] | × ideal R1 → fd | in95 R1 → fd | rank R1 → fd |
|---|---|---|---|---|---|---|---|
| < 15 | 148 | 0.50 / 0.53 / 0.50 | 0.090 → 0.075 | 0.93 [0.75–1.35] | 1.8 → 1.6 | 0.97 → 0.97 | 0.52 → 0.51 |
| 15–40 | 157 | 0.85 / 0.85 / 0.84 | 0.044 → 0.043 | 1.00 [0.73–1.38] | 1.9 → 2.0 | 0.97 → 0.98 | 0.51 → 0.47 |
| 40–100 | 151 | 0.95 / 0.96 / 0.95 | 0.016 → 0.017 | 1.07 [0.82–1.40] | 2.0 → 1.9 | 0.97 → 0.94 | 0.53 → 0.46 |
| ≥ 100 | 384 | 0.995 / 0.995 / 0.997 | **0.0067 → 0.0102** | **1.54 [0.98–2.27]** | **4.2 → 7.5** | 0.96 → 0.95 | 0.50 → 0.41 |

- **Loud floor 0.0103 bins** (CI 0.0096–0.0110): 1.54× R1 (CI 1.46–1.63), 0.26× XS-late-cool
  (0.039, same WDM net without features).
- By SNR band (100–200 / 200–400 / 400–800 / ≥ 800): 0.0127 / 0.0107 / 0.0094 / 0.0078 bins,
  against R1's 0.0100 / 0.0063 / 0.0048 / 0.0037. It falls ×1.6 against R1's ×2.7: flatter,
  more like a precision limit.
- Detection 732 of 840 against 740 (10 found only by fd, 18 only by R1); in95 0.95–0.98; no
  bias in bins (+0.0007 ± 0.0054). Loud rank 0.41: the truth sits a few tenths of a width low.

**Against the MCMC (four windows, 128 source–parameter pairs).**

| | R1 | fd, x | fd, y |
|---|---|---|---|
| loud f₀ (SNR ≥ 500), median × MCMC | 6.3 | 11.4 | 13.8 |
| loud A, sky, orientation, median × MCMC | 5.0 | 6.1 | 5.6 |
| MCMC median inside the flow's 95% / 68% | 125 / 108 | 126 / 109 | 125 / 106 |
| sources missed (> 50% of draws > 1 bin off) | 6 | 6 | 5 |

- A, sky and orientation stay 3–9× with fp32 inputs and fp32 PLE coordinates, so the limit is
  set after the embeddings: the bf16 backbone, the loss's insensitivity (F6), or training.
- R1's biased SNR-235 source (F31) is inside the 95% now; the new outlier is the SNR-1366
  source's sin ι.
- The crowded SNR-64 source (1.1 bins from SNR 819) is worse: 27× in f₀, 22–61× elsewhere.
  Masked sources are missed as before.

**Schedule-free's average lags the training weights** (flow loss on 8192 identical samples,
paired):

| | flow loss | paired difference |
|---|---|---|
| x (evaluated) | 0.4623 | x − y = **+0.0118 ± 0.0004** |
| y (training) | 0.4504 | y − R1 = +0.0116 ± 0.0006 |
| R1 | 0.4389 | x − R1 = +0.0234 ± 0.0006 |

- At constant lr, x is the uniform average of z over the whole run. Both runs show F10's late
  loss drop at 75–90% of the run (R1 epochs 850–900, −0.034; XS-fd 950–1100, −0.042), so x
  still carries pre-drop weights.
- x's held-out loss sat 0.014 below the training median before the drop, 0.009 above it at
  epoch 1000, and was catching up at the end (−0.022 over the last 100 epochs).
- XS-fd's training loss was 0.010 below R1's over epochs 50–850; its drop came 100–150 epochs
  later; at the end it was still falling at −0.058 per doubling of steps.
- y is not sharper at the loud end (13.8× against 11.4× in f₀): the lag explains the loss gap,
  not the floor.
- R1's cooldown (as in C1, F23) took its floor from 0.010 (F30) to 0.0067. XS-fd's x and y are
  both constant-lr models. The y scorecard (14401076) will show this on all 840 sources.

**Not the embedding cast.** The bf16 cast of the source embedding limits x_f to 1.0×10⁻³ bins
along the embedding's gradient, ten times finer than the floor; worst case 0.03 bins. A, sky
and orientation: 1–2×10⁻⁴ in coordinate units.

**Cost.** 52 s per epoch against 38 s for the WDM nets (+37%), mostly Muon's Newton–Schulz on
71M backbone parameters, which XS's small network does not amortise. B-fd: 1614 s against B1's
1536 (+5%).

**B-fd at epoch 40** (14326117): training loss 0.571 against B1's 0.547. The gap is closing
(0.053 at epoch 10, 0.037 at 30, 0.024 at 40), as XS-fd's did before it overtook R1. Gradient
norm rising (0.63 → 0.77).
