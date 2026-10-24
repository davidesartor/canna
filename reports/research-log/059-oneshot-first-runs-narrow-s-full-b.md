# F59 (2026-10-24): the oneshot code's first runs; narrow-S is the best narrow-band model, full-B's f₀ is stuck at 0.4–0.6 bins

- **narrow-S beats XS-stft-h with a 2.2× smaller network:** 779 of 840 sources found (646),
  f₀ 0.026–0.038 bins below SNR 40 (0.056–0.133), and at SNR 15–100 every parameter within
  0.8–3.3× of the MCMC in the band medians (XS-stft-h 2–54×); the three sources there above
  5× sit within 1.3–4.1 bins of a neighbour. The loud sources' f₀ is 2× wider (0.021 against
  0.010 bins).
- **full-B regressed on f₀ against B-stft-h:** 0.41–0.57 bins at every SNR (B-stft-h
  0.033–0.133), 10–430× the ideal, 38–164× the MCMC above SNR 40. φ₀ follows (36–70× at SNR
  ≥ 40, B-stft-h 0.7–2.0×); sky, A, ψ and ι stay within 1–2.7× below SNR 300, like B-stft-h.
- **The floor scales with the window, not the physics:** f₀ widths are 1.3–1.9 × 10⁻⁴ of
  full-B's 3072-bin window and 2.2–4.0 × 10⁻⁴ of narrow-S's 96-bin one, the same at low and
  high f₀ (0.44 against 0.47 bins). B-stft-h's 4096-bin window gave 0.033 bins, so a
  coordinate-resolution floor in the new code, not the band.
- **Two bugs found:** TARP's reference point is the truth itself, so every TARP cell is
  −0.50, and two init keys collide (both are the RNG collision below).

**Runs** (oneshot 3e5bdd4, tagged `oneshot-runs-3e5bdd4`; the branch has since been rewritten,
with no change to training, the network or the scorecards). The defaults of `narrow-S` and
`full-B` (batch 4096 = 256 observations × 16 draws, bf16 backbone, Muon lr 1e-2, AdamW 1e-3,
time_power 1, late_time_mix 0.5) plus `LISA_LOG_INTERVAL` from the [training
overhead](../benchmarks/training-overhead/README.md) report (10 000 and 5000; it must divide
`total_steps`). TREX A100 80 GB PCIe:
- narrow-S 15565104, 650k steps, 14 h 04 min, 0.077 s/step (the report's SXM4: 0.073);
- full-B 15565105, 240k steps, 27 h in two chunks, 0.40–0.50 s/step (0.363). Its
  self-requeue at the 24 h limit worked on TREX (first time), resuming at epoch 42.
- Post: eval 3 min, scorecard 7–8 min, the narrowband MCMC 80 chains per band (narrow 6.5 min
  each, full 52–93 min), `mcmc report`.

Outputs `29-lisa-narrow-S`, `30-lisa-full-B` (runs `misty-meadow-narrow-S`,
`lucky-falcon-full-B`; corner pages in each run's `corner/` → `eval/<epoch>-average-32/corner/`,
with `losses.pdf`, `lambda.pdf` and `validation.pdf`, drawn by `canna.lisa.plots`), chains `mcmc/eval-ad8a87a575`, `mcmc/eval-3867916973`, tables
`scorecards/oneshot_first_runs.{py,txt}`. Held-out x 0.8129 (narrow-S) and 0.4527 (full-B),
flat over the last 10% (+0.0005 and +0.00005 per epoch).

**Comparisons are by SNR band, not per source:** the oneshot problem has a new sky prior (the
galactic mixture), so its eval injections differ from fml's. The predecessors are B-stft-h
(F56, full band, 512 × 8 × 8) and XS-stft-h (narrow band, the same 512 × 8 × 8 network: on fml
"XS" named the band, not the size).

**1. f₀ per source** (all sources; found = width < 1 bin; width and × ideal over found sources):

| SNR | narrow-S found | width | × ideal | XS-stft-h width | full-B found | width | × ideal | B-stft-h width |
|---|---|---|---|---|---|---|---|---|
| 0–15 | 0.68 | 0.038 | 0.7 | 0.133 | 0.74 | 0.568 | 10 | 0.133 |
| 15–40 | 0.87 | 0.026 | 1.4 | 0.056 | 0.91 | 0.508 | 22 | 0.048 |
| 40–100 | 0.99 | 0.024 | 3.0 | 0.026 | 0.88 | 0.481 | 57 | 0.037 |
| 100–300 | 0.99 | 0.023 | 7.7 | 0.020 | 0.99 | 0.460 | 139 | 0.040 |
| ≥ 300 | 1.00 | 0.021 | 23 | 0.010 | 0.98 | 0.407 | 430 | 0.033 |
| found | 779 / 840 | | | 646 / 840 | 771 / 840 | | | 215 / 240 |

- narrow-S below SNR 15 is overconfident: 0.7× the ideal width, the truth in its 68% / 95% for
  0.47 / 0.78 of sources (XS-stft-h 0.64 / 0.91). In the validation its quietest slot's f₀ ranks
  have 0.36 in the middle half (too narrow), its loudest slot's 0.67 (too wide).
- full-B is conservative everywhere: in68 0.79–0.92, f₀ middle-half 0.68–0.76 in every slot.

**2. Flow / MCMC width, median per SNR band** (40 eval sources per run; [min–max] in the table
file):

| SNR | run | f₀ | A | λ | β | ψ | sin ι | φ₀ |
|---|---|---|---|---|---|---|---|---|
| 15–40 | narrow-S | 1.03 | 1.08 | 1.09 | 1.12 | 0.83 | 1.04 | 0.77 |
| 40–100 | narrow-S | 3.25 | 1.84 | 1.96 | 1.94 | 1.81 | 1.60 | 2.59 |
| ≥ 300 | narrow-S | 31 | 5.8 | 7.9 | 6.9 | 5.6 | 6.5 | 11.5 |
| 15–40 | full-B | 16.7 | 0.92 | 1.28 | 1.18 | 0.97 | 0.95 | 2.48 |
| 40–100 | full-B | 38 | 1.16 | 1.65 | 1.44 | 1.19 | 1.17 | 40 |
| 100–300 | full-B | 70 | 1.67 | 2.13 | 1.78 | 1.25 | 1.18 | 36 |
| ≥ 300 | full-B | 164 | 2.9 | 7.6 | 4.8 | 3.2 | 2.6 | 70 |

- B-stft-h's MCMC sources at SNR 15–100: f₀ 1.4–4.2×, φ₀ 0.7–1.4×, sky 1.1–2.2× (F56). XS-stft-h
  at SNR 15–100: f₀ 2.8–93×, sky 2.5–36×.
- The MCMC median is inside the flow's 95% for 95–100% of sources in every parameter, both runs:
  nothing is biased.
- **Per source the medians hide two failure kinds,** loud sources and close pairs. Sources with A,
  sky, ψ or ι above 5× the MCMC: narrow-S 17 of 40, all at SNR ≥ 121 or within 4.1 bins of
  another source (its four sources share 96 bins, median neighbour 10 bins); full-B 11 of 40,
  all at SNR ≥ 146 except an SNR-79 source 5.1 bins from an SNR-261 one (sky 133–136×, its
  flow mixing the two: the gb-median page) and an SNR-7 source. The close pairs are
  [F29](029-crowding-detection.md)'s masking, not new.
- φ₀ tracks f₀: a 0.45-bin f₀ spread makes the phase at t = 0 uncertain by about π × 0.45 ≈ 1.4
  rad. So full-B's φ₀ is a consequence of its f₀, not a separate failure.

**3. The floor.**
- Found widths over the window: full-B 1.3–1.9 × 10⁻⁴, narrow-S 2.2–4.0 × 10⁻⁴.
- full-B's window is 3070–3072 bins at every f₀: it is sized for 12 mHz, so a 0.1 mHz source
  gets the same window, and its width is the same in the low and high f₀ halves (0.44, 0.47
  bins).
- The B-stft-h → full-B differences that could set it: the learning rates (Muon 1e-3 → 1e-2,
  AdamW 3e-4 → 1e-3), time_power 3 → 1 with late_time_mix 0.5, batch 256 → 4096 samples, and
  the network's new f-conditioned one-way data stream and RoPE on the shared frequency axis.
- [F26](026-b1-3-bin-floor.md) had a similar window-relative floor (~2 × 10⁻³ of x_f) that RoPE
  broke ([F44](044-b-rope-breaks-3-bin-floor.md)).

**4. Two bugs, one cause.** JAX's default partitionable threefry makes `jr.fold_in(key, i)`
equal `jr.split(key, n)[i]`.
- `validate.validate_injection` draws the truth from `split(key, 4)[1]` and TARP's reference
  from `fold_in(key, 1)`: the same key, so the reference *is* the truth (distance 0 in all 1000
  injections). Every TARP cell is −0.50 in both runs. A new key stream fixes it (`key_tarp`, in
  b6e0218 on the rewritten branch).
- `LisaFlow.__init__` splits its key into 10 and gives `hat_embed` `fold_in(key, 1)` and the sinks
  `fold_in(key, 2)`, the keys of the second and third modules. The modules differ in shape and
  role, so this is likely harmless.
- The joint log-likelihood rank is +0.50 with nothing in the middle half (the truth fits better
  than every draw) in both runs. That is real: nearly every window holds a source above SNR 100,
  whose draws are 5–160× the MCMC.

**What follows.**
- **Narrow band:** narrow-S is the narrow-band model. Left: loud f₀ (31× MCMC, 2× XS-stft-h's
  width), loud phases and sky (6–12×), and low-SNR f₀ overconfidence.
- **Full band, cheapest first:**
  - the [velocity probe](../architecture/data-dependence/) on full-B's x_f near s → 1;
  - a full-B run with time_power 3, B-stft-h's learning rates, or both;
  - a window sized per f₀, which would also cut the step.
- **Fix the TARP key** before the next scorecard.
