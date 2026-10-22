# F39 (2026-10-13): SF-NorMuon's and ScheduleFree+'s fixes, integrated (renumbered from F38, which a parallel session also used)

`src/canna/lisa/schedule_free.py` replaces `optax.contrib.schedule_free`; XS-fd and B-fd use it
with:
- **Decay on z** (SF-NorMuon): z ← z − ηλz + base step, λ = 0.05, at each group's lr (Muon's or
  AdamW's), matrices and embedding tables only. The base optimizer no longer decays; without
  `--schedule_free`, decay stays optax's decoupled one.
- **Inner momentum** (both papers): Muon 0.8 (SF-NorMuon's), AdamW b1 0.9 (was 0).
- **Averaging weights t^r lr², r = 1** (ScheduleFree+).
- **Warm-started averaging** (ScheduleFree+): x follows z for the first 2 × warmup_steps.
- **β from 0.9 to 0.965**, log-linear in t over total_steps (ScheduleFree+).
- **x kept in the state**: a varying β needs it, and it removes F34's float32 reconstruction
  error.

**Not taken over** (corrected 2026-10-13 after reading ScheduleFree+ in full):
- **Polyak step** (Eq. 12, Algorithm 1): needs the loss minimum f*, which Algorithm 1 sets to
  0. Flow matching's irreducible loss (~0.4 on XS) would then scale steps with the whole loss.
- **AdamC's fully decoupled decay** (γ_t² λ z, λ 5–50; Eq. 14): "necessary when using the
  Polyak step size". With a set lr, our γλz differs only through the warmup.
- **NorMuon's row-wise normalisation**: a different base optimizer, not a fix (~12% faster
  convergence per SF-NorMuon).
- **Inverse gradient-L1 step weighting** (Eq. 10): left out by mistake. It is not tied to the
  lr-free step (the paper also runs "ScheduleFree 1/L1" with a grid-searched lr, Fig. 8). Step
  ∝ 1/EMA‖g‖₁ (EMA 0.9) counters gradient-norm drift, which our runs show: 0.30 → 0.59 late in
  XS-fd, 0.63 → 0.80 by epoch 48 of B-fd.

**Checks.**
- `tests/lisa/test_schedule_free.py`: with fixes off it matches `optax.contrib.schedule_free`
  step for step in y and x; decay holds a constantly pushed z at u/(ηλ); x follows z until the
  warm start; r = 1 brings x to late iterates faster; β is monotone. Full suite passes (844).
- 400-step laptop run (XS-fd, warmup 50): z − x stays 0 for the 100 warm-start steps, then
  opens; backbone weight RMS holds at 0.019.

Checkpoints now carry x too: ~0.3 GB more on XS.
