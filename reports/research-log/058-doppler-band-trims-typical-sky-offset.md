# F58 (2026-10-22): the Doppler band trims the typical sky offset but not the large ones; the individual biases are not a missing Doppler reference

- **The typical λ offset halves,** in both Doppler regimes, and population calibration tightens,
  at +3% step time and no cost elsewhere.
- **The individually biased skies stay:** q0.10 s2 λ 2.9 MCMC widths off, the loudest source's
  sin β 21.9; sky widths against the MCMC 1.4–2.3× (predicted ≤ 1.2×).
- **Left as causes:** the antenna-pattern part of the response, the amortisation error of one
  network over 0.1–12 mHz, or a correction this network size and training length cannot learn.
  A cause-independent remedy: a short Fisher-preconditioned MALA polish of each flow draw.

**Not in the current code:** the Doppler bands are at `origin/fml` 6e2f01e
([F57](057-b-stft-hd-doppler-bands.md)).

**Run** (commit 6e2f01e). B-stft-hd 15298962 (10 h 44 min, 210 s/epoch, 7.6 GiB, 84.5M
parameters; post 40 min). Output `28-lisa-B-stft-hd`. Held-out x 0.5063 (B-stft-h 0.5011).
Tools: `lisa_checks/sky_by_doppler.py at 05393cd` (offsets per Doppler-index band),
`lisa_checks/sky_offsets.py at 05393cd`, the MCMC tables, `outputs/scorecards/compare_ht.py`
and [flow-vs-reference](../posterior-eval/flow-vs-reference/) (phase combinations, width ratios).

**1. Sky offsets by Doppler modulation index** (sources above SNR 15; |z| = |flow median −
truth| in flow widths, ecliptic):

| run | index < 1 rad (17): median \|z\| λ, sin β; # > 1.5 | index ≥ 1 rad (17): median \|z\| λ, sin β; # > 1.5 |
|---|---|---|
| B-stft (22) | 0.53, 0.55; 3 | 0.60, 0.71; 2 |
| B-stft-h (26) | 0.46, 0.50; 1 | 0.67, 0.76; 6 |
| **B-stft-hd (28)** | **0.30, 0.40; 0** | **0.28, 0.87; 4** |

- sin β does not move. Outlier counts among 17 are noisy (B-stft and B-stft-h, differing only in
  the phase coordinates, give 2 and 6), so 6 → 4 is not a clear change.
- **Population calibration (40 eval sources):** above SNR 100 mean z −0.08, −0.02 (B-stft-h
  −0.34, +0.24), rms 0.53, 0.75; at SNR 15–100 rms z in β 0.90 (B-stft-h 1.43, one outlier gone).

**2. Against the MCMC.**
- **q0.10 s2** (0.76 mHz, index 2.4): λ still 2.9 MCMC widths (2.0 of the flow's) off, as in
  B-stft-h.
- **q0.10 s0:** sin β now 1.1 MCMC widths off (B-stft-h 0.2).
- **Loudest** (0.16 mHz, index 0.5): sin β 21.9 MCMC widths off (B-stft-h 12.4; 0.9 and 0.6 of
  the flow's width); f₀ 60× the MCMC (172×), φ₋ 52× (101×).
- **Sky widths,** q0.10 s0–s2: λ 1.6–2.3×, sin β 1.4–1.7× (B-stft-h 1.6–2.2×, 1.4–1.5×).
- **Mid-SNR pinned φ±:** 2.1–3.9× (B-stft-h 1.8–2.3×).

**3. Everything else unchanged:** 0.93–1.15× B-stft-h per source in every parameter; f₀
1.07–1.23× B-stft's, 213 found (215); in68 0.90, in95 1.00.

**F57's predictions.**
1. Doppler-dominated outliers ≤ 2 of 17: **missed** (4); MCMC sky widths ≤ 1.2×: **missed**.
2. No change below 1 rad: **partly wrong**; the offsets shrank there too (0.30, 0.40, no
   outliers), so part of the gain is not specific to the Doppler regime.
3. f₀ and the phases unchanged: **held.**
4. **The fallback applies:** the large per-source offsets are not a missing Doppler reference.
