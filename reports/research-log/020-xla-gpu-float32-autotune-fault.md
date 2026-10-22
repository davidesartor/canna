# F20 (2026-10-05): compiled float32 is wrong because of an XLA GPU compilation fault, triggered by its default autotuning
- Explains F14 and F18. One fixed program is right or wrong depending only on
  `--xla_gpu_autotune_level`: 0–3 right, 4 (default) and 5 wrong. The fault is in XLA's GPU
  compilation, not the model code; bf16 is unaffected.
- The laptop workaround (level 3 + `highest`) is **not enough on the A100** (F22); jobs now use
  level 0.

Laptop RTX 2000 Ada, tiny float32 test network; scripts in
`outputs/lisa-XS/eval-tools/fp32bug/`.

**Reproducer: one velocity evaluation.** Eager GPU vs CPU: 2.9×10⁻⁴; jitted: **0.595**, on
velocities ≤ 0.45. `same_program.py` hashes the lowered StableHLO (625 ops, no custom calls);
the hash is identical in every run below, only `XLA_FLAGS` differ. Error vs CPU:

| setup | default level (4) | level 3 | level 0 |
|---|---|---|---|
| jax 0.11.0, CUDA 12 (cuBLAS 12.9.2) | 0.595 (two fresh processes; also with no GPU preallocation) | 2.9×10⁻⁴ | 3.0×10⁻⁴ |
| jax 0.10.1, CUDA 12 | 0.595 | 3.0×10⁻⁴ | – |
| jax 0.10.1, CUDA 13 (cuBLAS 13.1.1) | 0.595, the same bytes as CUDA 12 | 2.9×10⁻⁴ | – |

A performance flag cannot change a fixed program's meaning (no undefined HLO behaviour for
these ops, no custom calls). Not machine-specific either (laptop Ada on driver 580; F18's A100
on driver 550 shows the same symptom), nor memory pressure. Repeated calls in one process give
the same wrong bytes.

**Bisection** (the exact faulty XLA step is not identified):
- Not the attention (right alone under jit; a hand-written one keeps the error, clearing F18's
  suspect), nor `scan`/`jax.checkpoint` (one plain block fails the same way).
- Program-dependent: all stages right when the jitted function returns every intermediate,
  wrong when it returns one. In a block, the modulation alone is right; its paths into a matmul
  are wrong.
- Turning off Triton GEMMs, cuBLASLt or command buffers doesn't fix it; `highest` doesn't help
  at level 4. At level 4 the x embedding's second Linear becomes a cuBLASLt matmul with a BIAS
  epilogue, but that layer compiled alone is right, and with cuBLASLt off the bug stays.

**Upstream:** [jax-ml/jax#35515](https://github.com/jax-ml/jax/issues/35515): float32 on an Ada
RTX 4090, jax 0.9.0, default level; its level-4 check rejects every candidate as "WRONG
RESULTS"; workarounds include `--xla_gpu_autotune_level=3`. Same machinery, not necessarily the
same bug: there it errors, here a wrong plan runs silently.

**bf16:** compiled vs eager on the GPU differ by 4.9×10⁻³, ~2 bf16 rounding steps. A100 bf16
scorecards match the CPU (F15), and so does the CPU fp32 scorecard (F21).

**Laptop workaround:** level 3 plus `JAX_DEFAULT_MATMUL_PRECISION=highest` (true fp32 matmuls,
not TF32): compiled fp32 matches the CPU to 2×10⁻⁷. `slurm/lisa-eval.sbatch` and
`lisa-scorecard.sbatch` set both for `--dtype float32`.
`test_fori_loop_transport_matches_the_unrolled_rk4[3]` still misses its 10⁻⁵ tolerance on the
GPU by 1.6×10⁻⁵: TF32, not the bug.

**Open:** confirm on the A100 (`lisa-scorecard.sbatch XS-late --dtype float32 --n_random 40`,
~3 min, must reproduce T3', F21); report upstream with `same_program.py` reduced so it no
longer needs canna.
