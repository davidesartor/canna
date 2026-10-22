# F38 (2026-10-13): B-fd's last read, epoch 45 of 200 — 10 bins wide, but no longer flat in SNR

- Loud median 9.5 bins, 3.3× B1's cooled final, but the width falls with SNR (slope −0.28)
  where B1's was flat: still training-limited.
- Calibrated and unbiased; loss about to cross B1's.
- Nothing failed the < 1 bin rule yet, and nothing showed it would pass.

**Run.** B-fd link 1 (14326117), checkpoint 45 (20.5 h in), hard-linked to
`outputs/B-fd-ep45/lisa-B-fd` on TREX. Scorecards of x (14404725) and y (14404726) with
`--n_random 50` as B1's, and the corner eval of x (14404727). Local copy and comparison in
`outputs/lisa-B-fd/ep45/` (`compare_bfd.py 45`). Caveat: the only B reference is B1's final,
cooled model (epoch 200); B1 was never scored mid-run.

**Loss.** B-fd minus B1 at the same epoch: +0.053 (10), +0.050 (20), +0.037 (30), +0.024 (40),
**+0.018 (45)**. B-fd falls 3.5× faster per epoch (−0.0019 against −0.0005 over epochs 40–45),
so it would cross B1 near epoch 50–55, as XS-fd crossed R1. Gradient norm still rising (0.78).

**Scorecard: the same 240 sources as B1's (F26).**

| SNR | n | median f₀ width (bins), B1 final / x / y | × ideal B1 / x / y | in68 B1 / x / y | rank B1 / x / y |
|---|---|---|---|---|---|
| < 15 | 40 | 5.0 / 342 / 331 | 98 / 6488 / 5073 | 0.90 / 0.78 / 0.70 | 0.49 / 0.53 / 0.53 |
| 15–40 | 38 | 3.8 / 16.7 / 13.5 | 197 / 668 / 595 | 0.84 / 0.95 / 0.92 | 0.38 / 0.48 / 0.49 |
| 40–100 | 47 | 2.9 / 10.5 / 10.5 | 361 / 1263 / 1164 | 0.89 / 0.94 / 0.96 | 0.43 / 0.51 / 0.53 |
| ≥ 100 | 115 | **3.2 / 9.5 / 9.9** | 2148 / 5756 / 5693 | 0.89 / 0.98 / 0.93 | 0.45 / 0.49 / 0.49 |

- **Width.** Loud median 9.5 bins (CI 9.0–10.5). No source found (< 1 bin) by x; y has one
  under 2 bins, B1 two. Faint sources are not localised (340 bins in a 4096-bin window, B1 5).
- **x vs y:** y/x = 0.98 [0.71–1.14] per source; unlike XS-fd's end (F33), x does not lag y in
  f₀ yet.
- **Slope of width vs SNR** (SNR ≥ 15): **−0.28 (x), −0.20 (y), B1 −0.05**. B1's was a flat
  resolution floor (F26); B-fd's shrinks 16.7 → 9.5 bins.
- **Calibration.** Unbiased (rank 0.48–0.53, B1 0.38–0.45) and conservative (in68 0.93–0.98
  above SNR 15). Found-and-loud offsets are a few bins; the mean is dominated by unlocalised
  faint sources (up to 2000 bins).
- **In flow coordinates.** 9.5 bins is 6.2×10⁻³ of x_f (1 unit = 1536 bins on B, F26), against
  B1's 2.1×10⁻³ and XS-fd's final 4.3×10⁻⁴; 0.8 of an f₀ PLE bin (12 bins on B, 0.19 on XS).
  XS-fd ended at 1/20 of a PLE bin; matching that on B gives ~0.6 bins, just inside the rule.

**Weight decay.** B-fd has the same effectively-zero decay as XS-fd (F34), so its weights grow
likewise, with an implicit uneven lr decay. The rising gradient norm (0.54 at epoch 6 → 0.78)
fits; nothing separates it from ordinary training.

**Epoch 48.** The user cancelled B-fd at epoch 48 (F35). Checkpoint 48 (loss 0.5580, 0.0155
behind B1) is on TREX, unscored. Checkpoint 45 is backed up in
`outputs/lisa-B-fd/ep45/checkpoints/45`.

**Reading.** It needs another 10× in width over 155 epochs, without the cooldown that gave the
WDM nets their last ×2–3 (F23, F31). Risk: F33's bf16-cast bound on the source embedding,
scaled to B (×64 in bins), is ~0.06 bins typical, ~2 bins worst case. If the next B run stalls
at 1–3 bins, check that first.

**Eval (corner pages, 14404727)** in `outputs/lisa-B-fd/ep45/corner/`.
- Loudest source (SNR 1668): f₀ centred, but λ has three or four modes, A a low tail, ψ and φ₀
  near prior, against a tight Fisher ellipse.
- Median source (SNR 61): f₀, sky and A on the Fisher ellipse with a few small outlier islands;
  ψ, φ₀ broad.
- The loud sources are still immature in every coordinate, as expected of a mid-run model.
