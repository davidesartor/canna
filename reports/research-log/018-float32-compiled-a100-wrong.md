# F18 (2026-10-04): float32 compiled LisaFlow is wrong on the A100 too, so T3 is void
- The XS-late scorecard with `--dtype float32` (job 13719350, 6 min) found 0, 1, 2 and 4%
  across the SNR bands, medians 2–5 bins off the truth; bf16 finds 46–99%.
- The cause is the GPU path: compiled float32 LisaFlow is wrong on both GPUs, so F14 is not
  laptop-specific. Every production (bf16) result stands.

**Evidence.**
- The 500k model in fp32 on the *CPU* (F5) was fine: all sources found, the loudest 6% narrower
  (0.197 vs 0.210 bins; corrected 2026-10-05 from "wider").
- On the laptop's Ada GPU (F14), jitted float32 LisaFlow was wrong, eager right.
- bf16 is fine: the A100 bf16 scorecard reproduces the CPU table (F15). In bf16
  `MultiStreamAttention` calls cuDNN flash attention; in float32 it goes to
  `implementation="xla"`, so XLA's GPU attention lowering or fusion is the first suspect.

**Consequences.** The bf16-output-noise hypothesis is still untested, and any fp32 step on a
GPU (the plan's step 2a) needs this fixed first.

**Next:** a float32 single-block attention, jit vs eager, on the laptop GPU (seconds); an fp32
scorecard on the CPU for a few injections.

**E3' is running:** job 13719349, started 4 Oct on e7a0864 with `save_opt_state: false`. Past
epoch 17 with checkpoints saving, so the quota fix works; 59.6 s per epoch, ETA 5 Oct.
