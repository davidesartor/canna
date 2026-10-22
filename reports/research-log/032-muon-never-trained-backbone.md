# F32 (2026-10-12): Muon never trained the backbone, bf16 rounded the loud data; a new recipe for XS and B

- Every run from 11000056 to R1 and B2 trained 89% of the parameters (all block weights) with
  Adam at lr 1e-4, not Muon.
- `LisaFlow`'s bf16 cast of the data rounds loud pixels by up to ~10σ: a candidate for the loud
  non-f₀ widths (F31).
- New recipe (6a0661c): whitened rfft data, PLE embeddings in fp32, schedule-free around a
  correctly routed Muon (98.7% of parameters). It changes several things at once.

**Muon routing.**
- `optax.contrib.muon` (0.2.8) without dimension numbers takes exactly the 2-D arrays.
  `MMDiT`'s `filter_vmap`-stacked block weights are `(blocks, out, in)`, so they went to Adam:
  **67.1M of 75.7M on XS-pos**.
- Muon got the embeddings and heads (where it is usually not used) and the stacked `(8, D)`
  block biases, orthogonalised across blocks.
- optax reads weights as `(in, out)`, equinox stores `(out, in)`, so Jordan's width scaling hit
  the wrong side of the 2-D weights it did take.
- So the WDM-era "Muon at lr 1e-4" and F9's small steps describe Adam.

**bf16 data rounding.** Error of `sinh(bf16(y)) − sinh(y)`, `y = arcsinh(WDM)`, over 2048 XS
prior windows, in noise σ:

| pixel | share | rms error | max |
|---|---|---|---|
| < 4σ | 74% | ≈ 0 | 0.03 |
| 4–30σ | 16% | 0.07 | 0.23 |
| 30–100σ | 7.1% | 0.53 | 1.6 |
| 100–300σ | 2.5% | 1.5 | 4.6 |
| > 300σ | 0.2% | 3.6 | 10.6 |

F26 ruled out the bf16 cell of the f₀ coordinate, not this. The frequency-domain stream reaches
~735σ per part on XS; T stays below ~5σ.

**Recipe** (6a0661c; WDM code tagged `wdm-r1`).
- **Data:** `preprocess` returns the whitened rfft window (real/imag of A, E, T, unit noise
  variance per part), same window geometry as the WDM era, so priors, windows and scorecards
  compare one to one. Two bins per token: 32 on XS, 2048 on B.
- **PLE** (arXiv:2203.05556) for every input scalar, each followed by one linear map, all fp32:
  - data: 64 bins uniform in arcsinh to 1000σ (0.24σ wide near zero, ~70σ at 300σ; quantile
    edges would lump everything above ~80σ);
  - source coordinates: 64 uniform bins, f₀ 256;
  - window frequency: 64 bins in log f (the old sinusoid of f in Hz spanned only 1e-4–0.08
    rad).

  y tokens get learned positions. Velocity head and conditioning in fp32, MMDiT in bf16.
- **Optimiser:** schedule-free (arXiv:2405.15682) around optax's Muon, which keeps its own
  momentum inside the wrapper (SF-Muon, AMUSE arXiv:2605.22432 eq. 4). Muon lr 1e-3,
  `consistent_rms=0.2`, bf16 momentum; AdamW (b1 0, b2 0.95, lr 3e-4) for encodings, token
  positions, last head layer and 1-D params; weight decay 1e-6, 2% linear warmup, clip 1.
  q/k/v, SwiGLU value/gate and adaLN shift/scale/gate are orthogonalised one by one; blocks
  have no biases. **Muon takes 71.3M of 72.2M (98.7%)**, asserted at startup.
- **Removed:** aux heads, `warmup_frac`, Welford reweighting, cooldown, `save_opt_state`,
  `PositionedLisaFlow` (the user's call). Loss is the plain velocity MSE.
- **Checkpoints/logs:** optimiser state kept (evaluated weights are the schedule-free average
  rebuilt from z), ~610 MB on XS; `load_trained` returns the average. Each epoch logs it on a
  fixed held-out batch (`heldout=`), plus the training loss at y and the gradient norm.
- **Checks:** full test suite passes on CPU; 300-step bf16 laptop run clean (peak 2.0 GiB,
  gradient norm ~0.35, clip mostly idle); on CPU the averaged weights move off y as they
  should.

**Confounded.** One run per rung changes data, embedding, optimiser and schedule and drops
three features at once, so a miss won't say which change caused it. Cheapest diagnostics:
held-out loss at x vs training loss at y, gradient norm, routing printout.
