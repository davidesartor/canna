# F34 (2026-10-13): schedule-free Muon ran with no weight decay; Muon's cost is Newton–Schulz in float32; sources now ordered

- XS-fd's weight decay was effectively zero, so the backbone grew 18–19× and z spread wider
  than the weights themselves.
- Muon's extra step cost is Newton–Schulz run in float32; bf16 NS (b9bb3ef, default) saves 9 of
  52 ms on XS.
- Sources are now ordered loudest first (b9bb3ef), at the cost of ~38% longer transport paths.

**Training weights y (scorecard 14401076, all 840 sources).**

| | loud floor (bins) | loud in68 | loud rank | detection |
|---|---|---|---|---|
| y (training weights) | 0.012 | 0.74 | 0.48 | 731 / 840 |
| x (average) | 0.0103 | 0.69 | 0.41 | 732 / 840 |

y is unbiased but wider than x: the averaging hides no sharper model, and neither weight set is
converged.

**The schedule-free integration** (`lisa_checks/sf_precision.py at 05393cd`, XS-fd checkpoint).
- Mechanics are correct: parameters and z stay float32, float64 only in scalars (NS
  coefficients, bias correction); the wrapper costs 0.7 ms per step over Muon.
- float32 averaging lands 1% of x's movement from float64 over 1.2M steps, with a +0.8% rescale
  (late on, 1 − c_t with c_t = 8.4×10⁻⁷ rounds to a value fixed for ~10⁵ steps). Minor.
- **Weight decay is the real problem.** optax multiplies 1e-6 by the lr (as PyTorch AdamW), so
  decay was 1e-9 per step. Muon's steps have fixed size, 2×10⁻⁴ RMS.
  - Backbone RMS grew **18–19×** (0.0255 → 0.46–0.49); R1 with Adam at 1e-4 grew 3–4×.
  - z sits 0.49 RMS from x, more than |x| = 0.37.
- Consequences: behind normalisations the effective step falls as weights grow, an implicit lr
  decay of up to ~19×, uneven by layer; and schedule-free averages iterates spread over a
  region larger than the weights.
- Schedule-free and Muon both rely on weight decay (AdamW 0.05–0.1 in the schedule-free
  experiments; 0.1 with Moonshot's RMS-matched lr). Without it, "no schedule needed" was never
  shown.
- **Fix for the next run (not yet in the configs):** decay ~1e-2 to 1e-1 in optax's convention.
  Equilibrium weight RMS ≈ 0.2·√(lr / 2λ): 0.045 at lr 1e-3, λ = 0.01; 0.014 at λ = 0.1.

**Step time, laptop GPU** (XS, batch 256): forward + backward 344 ms (42%), schedule-free Muon
326 ms (40%), Muon alone 325 ms, AdamW alone 37 ms (4.6%), batch 1.3 ms.
- Newton–Schulz is ~1 TFLOP per step on 71M parameters, a quarter of the network's FLOPs. optax
  runs it in the momentum's dtype, which follows the float32 gradient even with bf16 momentum
  storage.
- The cost scales with parameters × width, not batch × tokens: visible on XS (256 × 36 token
  rows, +37%), not on B (2048 tokens, +5%).
- `newton_schulz_dtype: bfloat16` (b9bb3ef, default) casts Muon's gradients first, as the
  reference Muon does.

**A100 timings** (jobs 14405794, 14405795; [`bench_step.py`](../benchmarks/step-time-memory/scripts/bench_step.py)). XS, batch 256, 32
tokens:

| part | time | share |
|---|---|---|
| batch generation | 0.6 ms | 1.5% |
| forward + backward | 26.5 ms | 61% |
| optimizer, schedule-free Muon, NS bf16 | **12.1 ms** | 28% |
| optimizer, schedule-free Muon, NS float32 | 21.2 ms | 49% |
| optimizer, Muon alone, NS float32 | 20.8 ms | 48% |
| optimizer, AdamW alone | 1.3 ms | 3% |
| **full step, NS bf16** | **43.4 ms** | |

- float32 NS reproduces XS-fd's 52 ms: 43.4 − 12.1 + 21.2 = 52.5. bf16 saves 9 ms; Muon's own
  ~12 ms remains, against R1's 38 ms Adam step.
- On B the optimizer is under 1% of the step (forward + backward 1550 of 1592 ms).

**Fewer tokens** (A100, bf16 NS, full step):

| | patch 2 | patch 4 | patch 8 | patch 16 |
|---|---|---|---|---|
| XS (32 / 16 / 8 / 4 tokens) | 43.4 ms | 36.7 ms | 33.3 ms | 31.1 ms |
| B (2048 / – / 512 / 256 tokens) | 1592 ms | – | **325 ms (4.9×)** | **177 ms (9.0×)** |

XS gains little (optimizer and batch dominate). On B, 200k steps would take ~18 h at patch 8
and ~10 h at patch 16, instead of ~88 h.

**Ordered sources (b9bb3ef).**
- `sample_physical` lists sources loudest first by own SNR, so targets, prior draws and
  injections are ordered alike. The geometry is one source's `Product` per slot; `Set` and its
  assignment code are gone.
- LisaFlow adds a learned slot embedding and a second head reconstructing x1, trained on
  per-slot squared ambient distance at `aux_weight: 0.1`.
- Scoring keeps `match_sources`, so scorecards compare with earlier ones and the MCMC.
- **Fixed labels lengthen transport paths ~38%** (mean squared path length per source, flow
  coordinates, 4096 XS windows):

| pairing | total | f₀ part | amplitude part |
|---|---|---|---|
| `Set`, nearest permutation (before) | 8.73 | 0.58 | 0.58 |
| SNR order | 12.04 | 0.67 | 0.29 |
| f₀ order | 12.04 | 0.26 | 0.66 |
| random labels | 12.44 | 0.66 | 0.66 |

- Most of the excess is in the angles, which no ordering shortens. Flow losses are not
  comparable with earlier runs' (a laptop smoke run starts at 1.08 against 0.78).
- Near-equal SNRs swap labels within one posterior (SNR known to ±1); f₀ is known to ~0.01
  bins, so f₀ order would rarely swap.
