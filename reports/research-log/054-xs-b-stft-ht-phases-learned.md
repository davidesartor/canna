# F54 (2026-10-19): XS-stft-ht and B-stft-ht learn the phases; B's φ₀ ± 2ψ within 2–3× of the MCMC at mid SNR, f₀ pays 1.3–1.45×

*Settled by [F56](056-phi0-learned-without-template.md):* the helicity coordinates did it; without the
template φ₀ is learned as well and f₀ costs 1.03–1.25× B-stft, not 1.3–1.45×.

- **φ₀ and the ψ–φ₀ ridge are read from the data,** for the first time since F39: on B at SNR
  17–78 the pinned φ₀ ± 2ψ is 2.1–3.4× the MCMC (B-stft 29–101×), φ₀ 0.9–2.7×, ψ 0.9–2.4×.
- **XS:** the pinned combination is 7.6–22× the MCMC (XS-stft 15–1469×).
- **The cost is f₀:** 1.27–1.45× B-stft's width per source, 213 found against 217; XS 1.2× at
  mid SNR, 27 fewer found. B's A, sky and ι 1.05–1.2× wider.
- **The loudest source keeps a gap:** φ₋ 72× on B, 22× on XS, the same floor as its sky.

**Runs** ([F53](053-xs-b-stft-ht-setup.md)). XS-stft-ht 14951673 (11 h 51 min, 35 s/epoch,
2.3 GiB; post 31 min), B-stft-ht 14951676 (13 h 26 min, 240 s/epoch, 13.2 GiB; post 1 h
02 min). Outputs `24-lisa-XS-stft-ht`, `25-lisa-B-stft-ht`. Held-out losses (XS 0.6230, B
0.5064) are not comparable with earlier runs: different coordinates. Tools:
`reports/diagnostics/draw-widths/compare_draws.py at 09abb90` (phase combinations),
`outputs/scorecards/compare_ht.py` (f₀), `reports/diagnostics/draw-widths/plot_widths.py at 09abb90`
(width ratios), [`velocity_probe.py`](../architecture/data-dependence/scripts/velocity_probe.py) with its
rotation test on the final checkpoints.

**1. The phases are read from the data.**

| | phase velocity / target | R², s < 0.5 | head rows / median | rotate data 0.3 rad → φ± velocity |
|---|---|---|---|---|
| every run F39–F51 | 0.003–0.005 | 0.000 | 0.06–0.14 | 0.001 |
| XS-stft-ht | 0.58–0.61 | 0.42–0.59 | 0.48–0.57 | 0.61–0.64 |
| B-stft-ht | 0.64–0.65 | 0.66–0.74 | 0.95–1.00 | 0.53–0.58 |

**2. Against the MCMC (flow width / MCMC width), B-stft → B-stft-ht.** Bold: the combination
the source's inclination pins.

| B | SNR | sin ι | φ₀ | ψ | φ₊ | φ₋ |
|---|---|---|---|---|---|---|
| loudest | 1668 | 0.96 | 2.6 → 1.8 | 1.8 → 1.8 | 2.1 → 1.8 | **2248 → 72** |
| q0.10 s0 | 78 | −0.40 | 32 → **2.7** | 1.7 → 2.4 | **101 → 3.4** | 29 → 2.6 |
| q0.10 s1 | 68 | 0.70 | 2.1 → 0.9 | 1.1 → 0.9 | 2.1 → 0.9 | **67 → 2.3** |
| q0.10 s2 | 61 | −0.89 | 1.4 → 0.9 | 1.1 → 0.9 | **65 → 2.2** | 1.6 → 0.9 |
| q0.10 s3 | 17 | −0.83 | 1.4 → 1.1 | 1.3 → 1.1 | **29 → 2.1** | 1.4 → 1.1 |
| faintest | 7.3 | −0.39 | 1.7 → 1.7 | 1.5 → 1.8 | 2.6 → 2.5 | 1.5 → 1.5 |

- **The ψ–φ₀ ridge is reproduced at mid SNR,** at 2–3.4× the MCMC's width.
- **Folded robust widths (the MCMC tables):** φ₀ and ψ at 0.7–2.1× for every B source; the
  loudest source's φ₀ 2.1× (B-stft 4.3×).
- **XS:** φ₀ 2.3–12× where the MCMC pins it (XS-stft 30–61×).
- **φ₀'s width per source (40 eval sources), against B-stft:** 0.09 at SNR 15–100, 0.11–0.18
  above. XS against XS-stft: 0.56 at mid SNR, 0.03–0.06 above SNR 100.

**3. The cost.**

| f₀, per source against B-stft | 0–15 | 15–40 | 40–100 | 100–300 | ≥ 300 | found |
|---|---|---|---|---|---|---|
| B-stft-ht | 1.30 [0.99–1.56] | 1.42 [1.16–1.61] | 1.27 [1.11–1.46] | 1.33 [1.14–1.54] | 1.45 [1.19–1.54] | 213 (217) |
| B-template ([F47](047-b-template-b-fisher-no-phi0.md)) | 4.6 | 7.3 | 9.2 | 8.0 | 9.7 | 209 |

- **B:** A, λ, sin β and ι 1.05–1.20× B-stft's per source; ψ unchanged. The loudest source's
  sky against the MCMC is unchanged (20, 22 against 18, 21). Calibrated: in68 0.84, in95
  0.99, rank 0.50.
- **XS:** f₀ 1.17–1.20× XS-stft's below SNR 100, 0.73× above SNR 300; 631 found against 658.
  The loudest source's f₀ 14 → 2.7× the MCMC, its sky 7.4 → 3.2–3.5×. One source regressed
  badly: q0.10 s2 (SNR 34), f₀ 4 → 51×, sky 2 → 24–48×.
- **The template costs far less here than in B-template** (1.3–1.45× against 7–9×).

**What made the difference.** B-template had the template residual with the old coordinates
and frequency-domain tokens, and never moved φ₀ (F47). B-stft had STFT tokens and neither of
the other two ([F51](051-xs-stft-b-stft-results.md), [F52](052-phi0-velocity-ignores-data.md)).
This run had all three: either the helicity coordinates are what the template needed (one
target instead of four copies, [F49](049-b-helicity-sky-calibrated.md)), or the STFT is; one
run cannot say which. The f₀ cost is most likely the template's, as in F47.
