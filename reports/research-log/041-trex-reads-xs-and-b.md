# F41 (2026-10-14): the TREX reads — XS's average changes nothing; B matches B1 at a ninth of the cost, against a flat ~3-bin floor

- XS-sfplus's x confirms F40: ~1.5× wider than XS-fd.
- B-sfplus x matches B1's final model in 10 h against 85 h, better calibrated, but still on a
  flat ~3.2-bin floor; the < 1 bin rule fails.
- B's floor sits at the bf16 embedding cast's worst-case bound: suggestive, tested by a float32
  scorecard (14505803).

**Fetched** and verified against TREX (143 files identical): XS-sfplus's full checkpoint, its
TREX scorecards of x and y (`scorecard_trex.npz`, `scorecard_training_trex.npz`) and corner
pages; B-sfplus's checkpoint 200, scorecards of x and y (`--n_random 50`, B1's 240 sources),
corner pages and logs. Local in `outputs/lisa-{XS,B}-sfplus`, symlinked as
`outputs/sfplus/lisa-{XS,B}-fd`. TREX's XS copy deleted (quota 9.0 → 8.0 GB); B's stays until
its float32 scorecard runs.

**XS: x does not change F40's verdict.**
- TREX's y scorecard agrees with the local one (674 against 673 found).
- x: 682 of 840 found (y 674, XS-fd 732, R1 740), loud floor 0.0151 bins (12× ideal), loud in68
  0.79.
- Per source, x/y is 0.99–1.04× in width; x finds 8 more and covers better. x/XS-fd is
  1.50–1.62× at every SNR (vs R1 1.41–2.42×).
- Against the MCMC (x): loud f₀ 21.7× (y 20.4×, XS-fd 11.4×, R1 6.3×), loud A, sky, orientation
  7.2×, 8 of 16 missed, coverage 127/128.
- Loudest source (SNR 1366): XS-fd resolved the ψ–φ₀ degeneracy as clean stripes and A tightly;
  XS-sfplus smears ψ and φ₀, has a long low-A tail and a longer A–ι ridge. The precision loss
  is broad, not only f₀.

**B-sfplus, the same 240 sources as B1 and B-fd:**

| | found (< 1 bin) | < 2 bins | median f₀ width (bins), SNR < 15 / 15–40 / 40–100 / ≥ 100 | slope vs SNR | in68 / in95 / rank, SNR ≥ 15 |
|---|---|---|---|---|---|
| B1 final (WDM, cooled, 85 h) | 1 | 2 | 5.0 / 3.8 / 2.9 / 3.2 | −0.05 | 0.88 / 0.99 / 0.43 |
| B-fd, epoch 45 (F38) | 0 | 0 | 342 / 16.7 / 10.5 / 9.5 | −0.28 | 0.96 / 1.00 / 0.49 |
| **B-sfplus x** (10 h) | 2 | 25 | 4.4 / 3.4 / 3.0 / **3.24** | +0.01 | 0.93 / 0.99 / 0.51 |
| B-sfplus y | 9 | 33 | 7.6 / 4.1 / 3.2 / 2.99 | −0.08 | 0.57 / 0.93 / **0.62** |

- x is 0.88 / 0.81 / 0.99 / 1.03× B1's width per source by SNR band, with rank 0.51 against
  0.43, in 10 h against 85 h thanks to 16-bin tokens (F34).
- The floor is flat in SNR (slope +0.01) at ~3.2 bins, as B1's was.
- y is sharper but biased (rank 0.62, in68 0.57): not usable as is.
- Diagnostics follow XS: weights 0.021 (epoch 8) → 0.0118, gradient 0.49 → 8.4; z − x bounded
  at 0.021; x ahead of y by 0.004–0.009 from epoch ~45; a loss step over epochs 60–80 (0.720 →
  0.699).
- Loudest eval source (SNR 1668): f₀ bimodal, ~20% of draws ~2.5 µHz (~160 bins) off, in λ and
  β too; A and ι follow the Fisher ellipse, wider; ψ, φ₀ near prior.

**bf16 cast of the source embedding** (`lisa_checks/embed_resolution.py at 05393cd`, F38's check). f₀
resolution after the cast:

| | along the embedding's gradient | worst case | the run's loud floor |
|---|---|---|---|
| XS-sfplus | 4.9×10⁻⁴ bins | 0.026 bins | 0.015 bins |
| **B-sfplus** | 0.094 bins | **2.7 bins** | **3.2 bins** |

On B the floor sits at the worst-case bound, and the f₀ PLE bin is 12 frequency bins wide (0.19
on XS). A float32 scorecard of B's x (14505803, same 240 sources) tests it directly.
