# F14 (2026-10-02): on the laptop GPU, compiled code disagrees with eager code
- `test_fori_loop_transport_matches_the_unrolled_rk4` fails on the laptop's RTX 2000 Ada (after
  the driver fix) and passes on the CPU. Don't trust laptop-GPU evals or training until this is
  understood.

**Bisection** (tiny float32 test network): eager on the GPU matches the CPU to 3×10⁻⁴; *any*
jitted version (the eval's `fori_loop`, or a jitted unrolled RK4) is off by 0.63 flow units,
~2/3 of the whole transport. Matmul precision `highest` doesn't change it. The network takes
the xla attention path here (float32), not cuDNN.

**TREX is not visibly affected:** A100 corner plots show loud-source f₀ and sky marginals as
sharp spikes on the truth; an error this size would smear f₀ by ~15 bins. `scorecard.py` on the
1M model on TREX also checks the A100: its ten eval rows use F12's keys and draws, so they must
reproduce F12's CPU table.

**Open:** a minimal laptop reproducer, then a JAX/XLA version check.
