# F49 (2026-10-18): the sky is calibrated and the loud sky is a bf16 floor; B-helicity launched

- λ, β are calibrated; they look off only where the flow is wider than the MCMC.
- The loudest source's wide sky matches a bf16 token-cast floor (sources above SNR ~250).
- `helicity` coordinates (c, φ₊, φ₋) remove the four-copy (ψ, φ₀) helix. `B-fd-helicity` runs
  them with the Fisher loss and `decay_adamw: false`.

User's asks: keep Fisher (B-fisher, F46); for/against the template (B-template, costly in
pipeline); why recent runs miss the ι-dependent ψ–φ₀ quasi-degeneracy; why λ, β look off.

**1. ψ and φ₀.**
- The data are linear in two mode amplitudes (F-statistic, helicity basis), c = sin ι (1
  face-on): a₋ ∝ A (1 + c)² e^{−i(φ₀ − 2ψ)}, a₊ ∝ A (1 − c)² e^{−i(φ₀ + 2ψ)}. Checked in jaxgb
  to 1e-15.
- Face-on measures φ₋ = φ₀ − 2ψ (φ₊ flat), face-off the reverse, edge-on both. The degeneracy
  is a ridge along φ₀ ± 2ψ = const, width varying with ι.
- Old coordinates ((ψ, ι) on a sphere, φ₀ on a circle):
  - Near face-on ψ is the azimuth and φ₀ = φ₋ + 2ψ winds twice: a helix. Invariance under ψ → ψ
    + π and (ψ, φ₀) → (ψ + π/2, φ₀ + π) gives four copies.
  - Over the copies the head's targets (cos φ₀, sin φ₀) and cos ι (cos ψ, sin ψ) average to
    exactly 0, and the velocity is uninformative until x_t picks a copy. φ₀'s gradient appears
    last, the one F48's decay erased.
- New coordinates (`LisaGB.polarisation: helicity`): c on [−1, 1], φ₊ and φ₋ on circles, still
  11 coordinates.
  - (ψ, φ₀) → (φ₊, φ₋) is 4 to 1 with the four symmetries as kernel: one copy, degeneracy
    axis-aligned. Prior stays uniform; physical columns stay (ψ, ι, φ₀), so MCMC, scorecards
    and tables are unchanged.
  - `flow_to_physical`: ψ = (φ₊ − φ₋)/4, φ₀ = (φ₊ + φ₋)/2. Corner pages fold samples onto the
    truth's copy (`eval.fold_orientation`); before, the MCMC covered 2–3 copies and B-long all
    four. Tests: `tests/lisa/test_helicity.py`.

**2. λ and β: calibrated, off only where too wide.** z = (flow median − truth) / flow width.
- 40 eval sources, B-rope/-template/-fisher/-long alike: above SNR 15 (34 sources) mean z −0.34
  to +0.19, rms 0.65–0.9, standard error ~0.2: no bias. Below 15, λ's mean z −0.55 to −0.87 on
  6 sources sharing one noise draw.
- Mid SNR (17–78, B-long): flow medians 0.1–1.1 MCMC widths from the MCMC's, which is itself
  0.3–1.4 from truth (noise). With the flow 1.1–1.6× wider, truth sits ~1 flow width off on a
  page: expected.
- Loudest (SNR 1668, 0.163 mHz): sky 30–44× the MCMC width, centre 1.3–1.7 own widths = 37–72
  MCMC widths off (B-rope 93–96); MCMC median outside the flow's 68% in sin β.
  - Not the f₀ correlation: the MCMC's sky-on-f₀ regression predicts 3–5 MCMC widths
    (`lisa_checks/sky_offsets.py at 05393cd`).
  - A flow conditioned on a coarser summary is calibrated but wider, centred on E[θ | summary],
    ~one own width from the full posterior's centre.

**Why coarse: bf16 tokens.** The float32 token embedding is cast to bf16. Token shift for a 1σ
(MCMC) shift, B-long, most affected token, max / rms over 512 components in bf16 ulps
(`lisa_checks/token_ulps.py at 05393cd`):

| source | SNR | f₀ | A | λ | sin β | ι |
|---|---|---|---|---|---|---|
| loudest | 1668 | 10.9 / 0.50 | 133 / 6.0 | **0.41 / 0.05** | **0.22 / 0.03** | 128 / 5.8 |
| q0.10 s0 | 78 | 1.6 / 0.48 | 2.1 / 0.39 | 1.1 / 0.28 | 1.0 / 0.27 | 1.7 / 0.42 |
| q0.10 s1 | 68 | 1.6 / 0.50 | 8.2 / 1.4 | 2.3 / 0.58 | 0.92 / 0.26 | 13 / 2.3 |
| q0.10 s2 | 61 | 4.6 / 0.66 | 7.3 / 1.3 | 1.1 / 0.27 | 2.1 / 0.25 | 8.6 / 1.4 |
| q0.10 s3 | 17 | 3.5 / 0.76 | 5.0 / 1.1 | 2.4 / 0.46 | 1.5 / 0.43 | 5.0 / 1.0 |

- The loudest is 5–10× below the rest, in the sky only: A, ι scale the whole pattern, the sky
  moves sidebands by ~1/SNR relative. bf16 resolves 2⁻⁸ ≈ 0.4%, so above SNR ~250 the sky is
  lost. Matches F45 (1–2× at mid SNR, 16–106× on the loudest); one rms ulp is 20–30σ there.
- φ₀ is not under the floor: on q0.10 s0 (MCMC 0.05 rad, flow 47×) a 1σ φ₀ shift moves the
  token 2.7 / 0.77 ulps, like f₀. That leaves F48's decay for φ₀.
- Untested: F42's float32 eval (no change for f₀) on the sky; a bf16-trained network may not
  read sub-ulp features anyway.
- Faintest (SNR 7.3): 25–59% of flow draws on the antipodal side; the MCMC is box-limited.

**3. The template.**

| loudest, flow / MCMC | λ width | sin β width | λ W1 | sin β W1 | f₀ width |
|---|---|---|---|---|---|
| B-rope (200k) | 66 | 106 | 58 | 124 | 56 |
| B-template (200k) | 57 | 95 | 45 | 84 | 460 |
| B-fisher (200k) | 54 | 79 | 45 | 108 | 69 |
| B-long (600k) | 30 | 44 | 27 | 75 | 52 |

- Weak measured case: B-template's gain (0.85–0.90× width, 0.7–0.8× W1) equals B-fisher's,
  B-long does more; no consistent mid-SNR gain; f₀ regressed 7–13×.
- For it: in y − h(x_t) the carrier is gone and the projection on ∂h/∂θ is the score Jᵀ(y − h),
  so sky and φ₀ errors are O(1) at any precision.
- Why B-template could not show it: x_t's f₀ is bins off for most of the path (F47), so the
  residual is junk until 1 − s ≲ 10⁻³; its six raw channels hit the same bf16 tokens.
- Cheaper form: from B-fisher's float32 J (`train.fisher_metric`), feed δ = (G + I)⁻¹ Jᵀ(y −
  h(x_t)), 11 numbers per source, to the source token. No new channels; +9% per step in
  training, J at every ODE step in sampling. Not in this run; user's call.
- Template-free test: the loudest source's eval in float32 (as F42). Only a positive result is
  informative.

**4. `B-fd-helicity`.**
- B-fd-rope-long (600k, warmup 12k) + `fisher_cap: 100`, `decay_adamw: false` (0.05 on z stays
  on Muon), `problem.polarisation: helicity`. ~35 h (209 s/epoch), `slurm/lisa-chain.sh
  B-fd-helicity 2 --n_random 50`. Reuses B's MCMC (`eval-27e6d96219`).
- Decay vs coordinates can't be separated if φ₀ returns; the probe's row norms will tell (F48).
  Held-out loss not comparable: use scorecards and MCMC tables.

**Predictions (before the result):**
1. Epoch ~50 (`srun --overlap`, [`velocity_probe.py`](../posterior-eval/flow-vs-reference/scripts/velocity_probe.py)): φ± velocity spread ≥ 20% of
   target (0.3–0.5% since F39), phase rows ≥ 0.5× the others'. Below 5%: cancel (saves ~30 h);
   both causes ruled out, next is analytic F-statistic extrinsics.
2. φ₀ ≤ 3× MCMC where it pins it (q0.10 s0, 47× now); ψ within 1–2×.
3. Loudest sky stays ≥ 20× (at most Fisher's ~0.8× over B-long). Below 10×: the bf16-token
   explanation above is wrong.
4. f₀ and found within 10% of B-long (221/240, 0.031 bins loud); calibration unchanged.
