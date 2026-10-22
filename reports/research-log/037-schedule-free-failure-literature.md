# F37 (2026-10-13): the literature on schedule-free failures — F36's runaway is documented for LLMs; no flow-matching reports

- No paper (search of 13 Oct) trains a diffusion or flow-matching model with a schedule-free
  optimizer; F36's failure is documented for language models.
- Papers in [the optimizer review](../literature/optimizers-schedules/000-takeaways.md):
  SF-NorMuon ([001](../literature/optimizers-schedules/001-apte26-sf-normuon.md), decay on z,
  the F36 mechanism), ScheduleFree+
  ([002](../literature/optimizers-schedules/002-defazio26-schedulefree-plus.md), r = 1,
  warm-started averaging), AMUSE ([015](../literature/optimizers-schedules/015-kim26-amuse.md),
  annealed β), EDM2 ([014](../literature/optimizers-schedules/014-karras23-edm2.md), weight
  growth, post-hoc EMA).

**Consequences for our code.**
- **Decay at y is the wrong place.** optax's `schedule_free` passes y to the base optimizer, so
  `optax.contrib.muon`'s decay acts at y, the placement SF-NorMuon finds unstable even at λ =
  0.05. Decay must act on z: our own wrapper (~40 lines, optax's plus −ηλz), or base decay off
  and z decayed in the wrapper.
- **Averaging weights.** For long runs, ScheduleFree+'s r = 1 with warm-started averaging, or
  AMUSE's annealed β, address the lag after the late loss drop (F33, F36).
- **Most precedent in diffusion:** constant-lr Muon with decay, plus an EMA (or post-hoc EMA)
  of the training weights.
