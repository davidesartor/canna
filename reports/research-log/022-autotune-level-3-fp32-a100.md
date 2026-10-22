# F22 (2026-10-05): level 3 does not fix fp32 on the A100; on the laptop, the real network and sampler are fine
- On the A100, level 3 removes the gross fp32 failure but leaves posteriors 2.4× narrower at
  the same distance from the truth: **overconfident** (in68/in95 0.45/0.88). F20's "level 3
  fixes the A100 too" is **refuted**.
- On the laptop, the real 768 network's compiled fp32 sampler matches the CPU at `highest`.
- So the A100 fp32 fault depends on the GPU or the 512 network, not the precision setting.
  Nothing in production depends on it: every run trains and evaluates in bf16.

**A100 (job 13790394, 3 min):** XS-late fp32 scorecard with `--xla_gpu_autotune_level=3` and
`JAX_DEFAULT_MATMUL_PRECISION=highest`, on T3''s 200 sources:

| SNR ≥ 100 | CPU fp32 (T3') | A100 fp32, level 3 | A100 fp32, default level (F18) |
|---|---|---|---|
| sources found, all SNR | 169 | 174 | 4 |
| f₀ width | 0.088 bins | 0.036 bins (0.41×) | garbage |
| median distance from the truth | 0.053 bins | 0.037 bins | ~2 bins |
| in68 / in95 | 0.85 / 1.00 | **0.45 / 0.88** | – |

CPU fp32 and A100 bf16 agree and are calibrated (F21), so A100 fp32 is still wrong, now subtly.

**Laptop GPU, real 768 network** (checkpoint 1000; `fp32bug/prod_velocity.py`,
`prod_sampler.py`):

| | one velocity call, max error / rms relative | full sampler, 256 draws × 32 RK4 steps, max \|draws − CPU\| |
|---|---|---|
| default level, default precision (TF32) | 1.7×10⁻² / 2×10⁻³ | 0.31 |
| default level, `highest` | 1.9×10⁻⁵ / 2×10⁻⁶ | 1.3×10⁻⁴ |
| level 3, `highest` | 1.7×10⁻⁵ / 2×10⁻⁶ | 1.4×10⁻⁴ |
| level 0, `highest` | 1.3×10⁻⁵ / 2×10⁻⁶ | – |

- No gross failure even at the default level: F20's belongs to the tiny test network's shapes.
- TF32's 2×10⁻³ per call moves single draws by up to 0.3 over 128 calls. That alone does not
  make the distribution wrong: bf16 is coarser still and calibrated.
- The A100 picks other kernels, and level 3 still lets it autotune.

**Next:** the eval and scorecard jobs now switch GEMM autotuning off for fp32 (`level=0`),
which matches the CPU on the laptop for both networks. One A100 run of `lisa-scorecard.sbatch
XS-late --dtype float32 --n_random 40` (3 min) must match T3'; if not, A100 fp32 stays
untrusted and fp32 checks stay on the CPU (88 s per injection).
