# F56 (2026-10-21): φ₀ is learned without the template; the helicity coordinates were the fix, not a phase reference

- **B-stft-h learns the phases as well as B-stft-ht:** the pinned φ± is 1.8–2.3× the MCMC at
  SNR 17–78 (B-stft-ht 2.1–3.4×, B-stft 29–101×); φ₀ and ψ 0.9–1.8× on every source.
- **On B the template's only effect was the f₀ cost:** without it f₀ is 0.64–0.89× B-stft-ht's
  per source, 1.03–1.25× B-stft's, 215 found.
- **B-stft and B-stft-h differ only in the phase coordinates,** so (φ₊, φ₋) is what made φ₀
  learnable. Corrects [F52](052-phi0-velocity-ignores-data.md)'s mechanism and
  [F54](054-xs-b-stft-ht-phases-learned.md)'s open attribution.
- **Left:** the loudest sources' phases (B 101×, XS 51×) and sky; a 1.2× loud-source cost in A,
  ι and sky.

**Runs** ([F55](055-xs-b-stft-h-setup.md), commit 3d87b22). B-stft-h 15212055 (11 h 26 min, 203
s/epoch, 7.4 GiB; post 39 min), XS-stft-h 15212052 (11 h 28 min, 34 s/epoch, 2.1 GiB; post 25
min). Outputs `26-lisa-B-stft-h`, `27-lisa-XS-stft-h`. Held-out x: B 0.5011, XS 0.6191 (24/25:
0.5064, 0.6230; same coordinates, so comparable). Tools as F54:
[flow-vs-reference](../posterior-eval/flow-vs-reference/) (phase combinations, width ratios),
`outputs/scorecards/compare_ht.py`, the MCMC tables, and
[`velocity_probe.py`](../posterior-eval/flow-vs-reference/scripts/velocity_probe.py) (with the rotation test)
on the final checkpoints.

**1. The phases are learned without the template.**

| B, flow / MCMC | SNR | sin ι | pinned φ±: B-stft (22) | B-stft-ht (25) | **B-stft-h (26)** |
|---|---|---|---|---|---|
| loudest | 1668 | 0.96 | 2248 | 72 | 101 |
| q0.10 s0 | 78 | −0.40 | 101 | 3.4 | **2.0** |
| q0.10 s1 | 68 | 0.70 | 67 | 2.3 | **2.3** |
| q0.10 s2 | 61 | −0.89 | 65 | 2.2 | **1.9** |
| q0.10 s3 | 17 | −0.83 | 29 | 2.1 | **1.8** |

- **B:** φ₀ and ψ 0.9–1.8× the MCMC on every source (MCMC tables 0.7–2.0×). Phase velocity
  0.64–0.65 of the target's spread, R² 0.69–0.75 at s < 0.5, head rows 0.98–1.01; a 0.3 rad
  rotation of the data moves the φ± velocities by 0.53–0.57, as B-stft-ht.
- **XS:** pinned φ± 4.1–17.8× (XS-stft-ht 4.3–22×, XS-stft 15–1469×); φ₀ 2.0–7.7× where pinned
  (XS-stft 30–61×); rotation response 0.57–0.62.

**2. What the template did** (per source, B-stft-h over B-stft-ht): f₀ 0.64–0.89×; A, sky, ψ, ι
and φ₀ 0.86–1.02×.

| f₀ per source / B-stft | 0–15 | 15–40 | 40–100 | 100–300 | ≥ 300 | found |
|---|---|---|---|---|---|---|
| B-stft-ht | 1.30 | 1.42 | 1.27 | 1.33 | 1.45 | 213 |
| **B-stft-h** | 1.03 [0.87–1.22] | 1.10 [0.84–1.25] | 1.15 [1.03–1.29] | 1.25 [1.08–1.37] | 1.03 [0.97–1.19] | 215 (217) |

- **On XS the template also helped the loudest sources:** without it their f₀ goes 2.7 → 13×
  the MCMC and sky 3.2–3.5 → 9.3–9.9× (XS-stft 14×, 7.3×); λ above SNR 300 is 1.47×
  XS-stft-ht's. XS f₀ 1.02–1.11× XS-stft's, 646 found (XS-stft 658, XS-stft-ht 631).
- **XS q0.10 s2 (SNR 34) regresses in both helicity runs** (sky 16–48× against 1.8–2.1× in
  XS-stft), so not the template.

**3. The cost of the coordinates** (per source, B-stft-h over B-stft): above SNR 100 A
1.21–1.22×, λ and sin β 1.03–1.18×, sin ι 1.16–1.23×; below SNR 100 within 1.08×. B-stft-ht
shows the same, so it comes with learning the phases (capacity or the chart), not the template.
XS above SNR 300: sky 1.22–1.25× XS-stft's.

**F55's predictions.**
1. Phases back to the prior: **wrong**, learned as with the template.
2. Template not needed: **right** (pinned φ± ≤ 6×, rotation ≥ 0.3). f₀ 1.03–1.25× B-stft's
   (predicted ≤ 1.1×: only at SNR < 40 and ≥ 300), 215 found (predicted ≥ 216).
3. f₀, A, sky, ι within 10% of B-stft: **missed for loud sources** (A, ι 1.2×).

**Correcting F52.**
- F52 said no data feature correlates with φ₀ without a coherent phase reference, so gradient
  descent gets no partial credit. B-stft-h has no reference input and learns, so that is not
  the cause.
- F49's argument fits: in (ψ, φ₀) the likelihood's symmetries make four copies of every
  posterior, and the head's (and early velocity's) target averages to zero over them, so the
  network has nothing to start from. In (φ₊, φ₋) there is one copy and a nonzero conditional
  mean.
- F52's measurements stand (the earlier φ₀ velocity ignored the data; one bin's phase does not
  correlate with φ±); only the mechanism was wrong.

**What follows.**
- The recipe is the -h configs: STFT tokens, helicity coordinates, no AdamW decay, no template.
- Left: the loudest sources' phases (B 101×, XS 51×) and sky (B ~20×, XS ~10×); the 1.2×
  loud-source cost in A, ι and sky; XS's mid-SNR f₀ and q0.10 s2.
- For XS's loudest sources a phase reference does help; the matched-filter phasors at the
  network's own x̂₁ remain the cheap form.
