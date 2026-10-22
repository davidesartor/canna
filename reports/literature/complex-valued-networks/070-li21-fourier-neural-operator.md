# Fourier Neural Operator for Parametric Partial Differential Equations

Li, Kovachki, Azizzadenesheli, Liu, Bhattacharya, Stuart, Anandkumar, 2021, ICLR ·
[arXiv:2010.08895](https://arxiv.org/abs/2010.08895)

**Evidence: A.** Peer-reviewed (ICLR 2021); 5192 citations in 6.0 yr (Semantic Scholar,
2026-10-19).

## What they do

The FNO learns maps between function spaces, such as PDE coefficients or initial conditions
to solutions. Each Fourier layer computes

v ← σ(W v + F⁻¹(R · F v)).

- F is the FFT over the spatial grid.
- R is a **complex-valued tensor** of shape k_max × d_v × d_v, one channel-mixing matrix per
  retained Fourier mode.
- Higher modes are truncated.
- W is a pointwise real linear map.

The kernel integral operator is thus a learned, mode-wise complex linear filter: a
convolution with a global kernel, parameterised in frequency space.

## What they find

- **Burgers:** relative L2 error 0.0149 at s = 256 and 0.0139 at s = 8192. The error is
  roughly resolution-invariant.
- **Darcy:** about 0.010, nearly an order of magnitude below the benchmarks.
- **Navier–Stokes:** FNO-3D reaches 0.0086 at ν = 1e-3. At ν = 1e-4 it reaches 0.082 with
  N = 10⁴ samples.
- Overall about 30–60% lower error than prior learned solvers.
- **Speed:** up to about 1000× faster than the pseudo-spectral solver (0.005 s vs 2.2 s on
  256²).
- **Zero-shot super-resolution:** trained at 64×64×20, evaluated at 256×256×80.

## Implementation details

- 1-D problems use k_max = 16 and width 64. 2-D problems use k_max = 12 per dimension and
  width 32.
- Because the kernel is real, "we impose conjugate symmetry", so the output stays real.
  Implementations use rfft and keep only non-negative frequencies.
- The mode-wise product costs O(n k_max), on top of the O(n log n) FFT. A uniform grid is
  required.
- Only the spectral weights are complex. Activations act in physical space on real values,
  so the network avoids complex non-linearities entirely.

## Caveats

- The paper does not discuss how the complex parameters are optimised or any numerical
  precision issues.
- The code fills that gap. The authors' repository and the `neuraloperator` library ship
  their own Adam/AdamW, whose complex branch is
  `exp_avg_sq.addcmul_(grad, grad.conj())`. The second moment is therefore |g|² per complex
  weight, invariant to rotating that weight's phase.
- That differs from stock PyTorch Adam, which steps complex parameters like their
  `view_as_real` pair: a separate second moment for the real and imaginary parts, so the
  preconditioner depends on the phase.
- Source: `neuralop/training/adamw.py` on GitHub, checked 2026-10-19.
- Mode truncation discards high-frequency content. That suits smooth PDE fields but is a
  low-pass assumption.
- Results are on smooth periodic PDE benchmarks. The paper offers no evidence that complex
  spectral weights beat equivalent real parameterisations, since that comparison is not run.
