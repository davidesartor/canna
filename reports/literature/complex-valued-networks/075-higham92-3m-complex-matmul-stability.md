# Stability of a Method for Multiplying Complex Matrices with Three Real Matrix Multiplications

Higham, 1992, SIAM J. Matrix Anal. Appl. 13(3), 681–687 ·
[doi:10.1137/0613043](https://doi.org/10.1137/0613043)

**Evidence: B.** Peer-reviewed (SIMAX); 51 citations in 34.3 yr (Semantic Scholar,
2026-10-19; OpenAlex gives 39).

## What they do

The paper analyses the **3M method**, Gauss's trick applied to matrices. For
(A + iB)(C + iD):

- T1 = AC, T2 = BD;
- Re = T1 − T2;
- Im = (A + B)(C + D) − T1 − T2.

That is three real matrix multiplications and five real additions, where the conventional
method needs four multiplications and two additions. The paper derives forward error bounds
for both methods and discusses when 3M is worth using. That depends on the relative cost of
real and complex arithmetic and on how it combines with fast (Strassen-type) or blocked
algorithms.

## What they find

- **Up to 25% fewer arithmetic operations**, "even for small dimensions".
- **3M is "less stable than conventional multiplication but stable enough to warrant
  practical use"** (abstract).
- **The real part** has the same error bound as the conventional method.
- **The imaginary part's** bound scales with (|A| + |B|)(|C| + |D|) instead of
  |A||D| + |B||C|. When the imaginary part of the product is much smaller than the operands,
  for example after near-cancellation, its relative error can be much larger.
- **The method is normwise but not componentwise stable.** That is acceptable for most
  uses, but not where small components must be accurate.

## Implementation details

- **3M is not the default everywhere.**
  - BLAS libraries offer it as separate routines, for example `cgemm3m`/`zgemm3m` in
    OpenBLAS and MKL.
  - cuBLAS exposes it as `cublasCgemm3m`.
  - The standard `cgemm` uses 4M, which on tensor-core hardware is 4 real GEMMs or an
    equivalent fused kernel.
- **Hardware context** (ours, not the paper's): complex matmul therefore costs about 3–4
  real matmuls of the same shape. Accelerator frameworks have no reduced-precision complex
  types.
  - XLA's `PrimitiveType` has only C64 (paired F32) and C128 (paired F64), with no
    F16/BF16 complex (`xla/xla_data.proto`, checked 2026-10-19).
  - PyTorch documents `cfloat`/`cdouble` as its complex dtypes, with complex support
    marked beta.
  - Mixed-precision complex layers are therefore usually written as real (Re, Im) pairs.

## Caveats

- Abstract read via OpenAlex. The error-bound statements are the standard results of this
  paper, also reproduced in Higham's *Accuracy and Stability of Numerical Algorithms*.
  The full text was not re-read for this note.
- 1992 hardware assumptions: where multiplications dominate the cost, 3M wins. On GPUs a
  4M kernel with better data reuse can be as fast, so the speed claim must be benchmarked
  per platform.
- The library/hardware paragraph comes from docs, not from the paper.
