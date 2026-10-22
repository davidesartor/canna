# F13 (2026-10-02): XS-late training and eval, first look (no numbers yet)
- The loss drop comes at epoch ~60, so it is not set by the aux schedule. Corner plots look
  like the 1M model's but cannot resolve the floor.

**Training.** 1M steps in 10 h 57 min, no NaN.
- Fast drop at **epoch ~60** (−3×10⁻³ per epoch, aux weight 0.35), vs ~190 for the 500k
  baseline and none by epoch 57 in E1 (aux 10%, uniform clock). It fits the network learning
  the late-path (fine-scale) structure that the warp up-weights.
- 500k → 1M lowered the loss 0.014, E0's pace. Values are not comparable to E0's: the warp
  re-weights the objective.

**Eval (corner plots only).** On q0.30, q0.60, q1.00 vs the 1M model: posteriors on the truth,
no visible collapse, the same sources found; XS-late's 2D sky contours somewhat more scattered.
- They cannot answer E2: each f₀ axis spans the 48-bin window, so 0.15- and 0.03-bin posteriors
  both draw as spikes. Floor (A), detection (B) and calibration (C) wait for `scorecard.py`
  (T1).
