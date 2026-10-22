# F9 (2026-10-01): the optimiser takes very small steps and never anneals
- Muon at lr 1e-4 (optax default width scaling): update RMS ≈ (0.03–0.045)·lr ≈ 4×10⁻⁶ per
  element per step. That is ~5× below AdamW at the same lr (RMS ≈ 0.2·lr) and ~200× below
  muon's reference lr ≈ 0.02.
- The lr is constant, so no cooldown banks the final loss.
- Neither is tested yet: this is E3; E5 adds a cooldown.
