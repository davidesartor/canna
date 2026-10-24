# F62 (2026-10-25): full-B-v32, velocity prediction in a float32 backbone, warm-started from full-B (setup)

**Why.** [F61](061-fp32-backbone-halves-f0-but-overconfident.md): half of full-B's f₀ floor is the
bf16 rounding of an absolute x̂₁, the rest an f₀ error learned in bf16. The user asked for the
offset fix, a warm start from run 30 (`lucky-falcon-full-B`, epoch 48) and float32 training.

**The run.** TREX job 15714731 at d5c447e, run `clever-falcon-full-B`, `full-B` with:
- `network.prediction: velocity`: the head outputs v and x̂₁ = x_t + (1 − s)·v, so any output error
  shrinks with the path left, as on fml; the loss is the same x-error over (1 − s); RK4 sampling.
- `bf16: none`: float32 backbone (autotune level 0, highest matmul precision).
- `init_from: lucky-falcon-full-B`: its averaged weights, except the head's output layer and λ,
  which velocity prediction redefines (kept at their fresh init); a fresh optimiser state.
- 24 000 steps, warmup 1200, log interval 500 (48 epochs), everything else full-B's defaults.
  Davide's [step-time report](../benchmarks/step-time-memory/README.md) put float32 full-B at
  8.4 s/step (23× bf16) on an A100-SXM4, so ~2.5–3 days in 24 h chunks.

**Predictions (written before the result).**
- f₀ on loud sources (SNR ≥ 100) at or below 0.15 bins (run 30: 0.44 bf16, 0.21 float32 eval),
  with in68 ≥ 0.6, i.e. not F61's overconfident float32 eval.
- If f₀ stays ≥ 0.3 bins, the floor is learned and not precision: next is time_power 3 and fml's
  learning rates.
- Other parameters within 1.2× of run 30's widths; the head relearns in the first epochs (loss
  above run 30's at first, then below).
