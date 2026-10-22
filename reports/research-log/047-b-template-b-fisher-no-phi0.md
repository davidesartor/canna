# F47 (2026-10-16): neither B-template nor B-fisher learns φ₀; along our path f₀ resolves too late for the phase to be readable

*Corrected by F48:* the schedule argument holds for B-template's x_t residual only. The plain
networks see the data at every s, and φ₀ was learned before F39. The likely cause is the weight
decay.

- **B-fisher** changes nothing measurable; **B-template** regresses f₀ to a ~0.33-bin floor.
- φ₀ is identical in all three runs: the network never moves it (velocity spread 0.5–1.3% of
  the target's, R² 0.000).

**Runs.** B-template (14706575 + post 14706576) and B-fisher (14706577 + post 14706578): 11 h
23 min and 11 h 43 min of training, post jobs 49 and 39 min. Local as
`outputs/19-lisa-B-template`, `20-lisa-B-fisher`. B-long (14706580) is at epoch 299 of 600,
held-out 0.6283, already below B-rope's final 0.6399. Figures: `outputs/plots/`
(`f46_plots.py`).

**Scorecards** (240 B sources, x; median f₀ width in bins; per source vs B-rope):

| model | found | 7–15 | 15–40 | 40–100 | 100–300 | ≥ 300 | rank / in95 | per source vs B-rope |
|---|---|---|---|---|---|---|---|---|
| B-rope | 215 | 0.91 | 0.055 | 0.037 | 0.037 | 0.036 | 0.51 / 1.00 | 1 |
| B-fisher | 216 | 0.93 | 0.054 | 0.036 | 0.036 | 0.036 | 0.50 / 0.99 | 1.00–1.08 |
| B-template | 209 | 1.37 | 0.42 | 0.33 | 0.32 | 0.33 | 0.47 / 1.00 | **6.8–8.3** |

- B-fisher's y is 0.79–0.91× B-rope at SNR ≥ 40, but y has been the less reliable output (F43).
  Its held-out (Euclidean) loss is 0.6568, worse than B-rope's.
- B-template's floor is flat in SNR, still calibrated.

**Against the MCMC** (B-rope / B-template / B-fisher):
- φ₀ identical to the first decimal: 47.3 / 47.5 / 47.2 (q0.10 s0), 4.7 / 4.7 / 4.7, 1.8 / 1.8
  / 1.8, …, 4.2 / 4.3 / 4.3 on the loudest.
- Loudest source's sky: 66–106 / 57–95 / 54–79× (B-fisher slightly narrower, one source).
- Everything else within noise of B-rope, except B-template's f₀ (13–460×).

**φ₀ diagnosis** ([`velocity_probe.py`](../architecture/data-dependence/scripts/velocity_probe.py), x of each run, 1024 training samples):
1. **The network never moves φ₀.** Predicted velocity spread 0.5–1.3% of the target's for cos
   φ₀ and sin φ₀, vs 47–98% for every other coordinate (chirp mass included); R² 0.000 at every
   s. The reconstruction head copies x_t's phase. Different networks give the same φ₀ per draw
   to a median 0.003–0.008 rad: the prior draws. All three runs.
2. **Not missing from the data:** a 0.3 rad φ₀ shift changes a SNR-900 source's whitened signal
   by 270σ; jaxgb's columns match ours (φ₀ enters as e^(−iφ₀)). Not a lack of incentive: φ₀ is
   ~30% of the flow-loss variance.
3. **Why:** the data's phase is only meaningful relative to a reference that turns by π per bin
   of f₀ offset (and with the sky's Doppler phase). All coordinates share one s, so f₀ resolves
   on φ₀'s schedule, too late for the phase:

   | 1 − s | B: f₀ of x_s off by | φ₀ of x_s off by | XS: f₀ off by |
   |---|---|---|---|
   | 10⁻² | 8.5 bins | 0.016 rad | 0.13 bins |
   | 10⁻³ | 0.85 bins | 0.0016 rad | 0.013 bins |
   | 10⁻⁴ | 0.085 bins | 0.0002 rad | 0.001 bins |

   When x_t's f₀ is sub-bin, its φ₀ is already at the truth. Before that the network would have
   to infer sub-bin f₀ from the data and demodulate internally, which it never learns.
   - B-template failed for this reason: its residual is computed at an x_t whose f₀ is bins off
     through most of the path, so it is junk for φ₀, and it hurt f₀.
   - B-fisher weights a velocity the network cannot predict.
4. **The same mechanism plausibly limits loud sources' sky** (16–106×): the sky enters through
   the Doppler phase too, and the sky velocity's R² drops to 0.09–0.13 at 1 − s < 0.01. A, ψ
   and ι, which also enter through amplitudes, are learned.

**What this changes:** not more weight or input on φ₀. Either resolve the phase after the
frequency, or leave it to the likelihood:
- **Per-coordinate schedules:** f₀ (and sky) converge earlier than A, ψ, ι, φ₀, e.g. s_f = 1 −
  (1 − t)⁶ vs s_e = 1 − (1 − t)³: at 1 − s_e = 10⁻², f₀ is at 1 − s_f = 10⁻⁴ (0.08 bins on B).
  A template at x_t, or F-statistic features, then carries the phase exactly when it is
  resolved. Still flow matching: each velocity is d/ds_k, multiplied by its own ds_k/dt at
  sampling.
- **Analytic extrinsics:** given (f₀, ḟ, sky), A, ψ, ι and φ₀ enter linearly through four
  F-statistic amplitudes (2004.08464); resample each draw's extrinsics from their exact
  conditional likelihood, with no training.
