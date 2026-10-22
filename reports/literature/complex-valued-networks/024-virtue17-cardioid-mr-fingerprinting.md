# Better than real: complex-valued neural nets for MRI fingerprinting

Virtue, Yu & Lustig, 2017, IEEE ICIP · [arXiv:1707.00070](https://arxiv.org/abs/1707.00070)

**Evidence: A.** Peer-reviewed (ICIP 2017); 197 citations in ~9 yr (Semantic Scholar,
2026-10-19).

## What they do

Regress tissue parameters (T1, T2, off-resonance B0) from a complex MR-fingerprinting
signal (length-500 complex time series simulated with the Bloch equations). Introduces the
**complex cardioid** activation and compares a complex fully connected network against real
networks fed 2-channel (Re, Im) input, and against other complex activations. Training on
100 k simulated signals; test on a numerical brain phantom with a linear B0 ramp
(−60…60 Hz), clean and with complex Gaussian noise (pSNR 40).

## What they find

NRMSE (%), T1 / T2:

| Model | Clean | Noisy |
|---|---|---|
| Nearest-neighbour dictionary | 10.63 / 39.78 | 12.21 / 40.38 |
| Real 2-channel, ReLU | 2.71 / 8.21 | 11.15 / **17.96** |
| Real 2-channel, 2× wider | 2.21 / 8.04 | 11.08 / 22.15 |
| Complex, cardioid | **1.42 / 4.34** | **9.40** / 20.98 |
| Complex, split sigmoid | 4.72 / 9.24 | 17.31 / 33.09 |
| Complex, siglog | 2.99 / 12.04 | 102.2 / 237.9 |

- Clean: cardioid halves the error of the real nets, and the 2×-wide real net does not close
  the gap ("cannot be explained away by the twice large model capacity").
- **Noisy: the advantage shrinks** (T1 9.40 vs 11.08) and **reverses on T2** (20.98 vs 17.96
  for the standard real net).
- Nearest-neighbour beats every network on B0.
- Activation choice dominates: cardioid ≫ split sigmoid and siglog (siglog collapses under
  noise).

## Implementation details

- **Cardioid**: `f(z) = ½ (1 + cos arg z) z`. Output phase = input phase; gain depends on the
  phase (1 on the positive real axis, 0 on the negative, smooth in between); reduces to ReLU
  on real inputs. Phase-*preserving* but **not phase-equivariant**: `f(e^{iα}z) ≠ e^{iα}f(z)`,
  because the gain depends on the absolute phase.
- 3 fully connected layers per output parameter, complex BN; Caffe with CR-calculus
  (Wirtinger) gradients, update `w ← w − α ∂L/∂w̄`. Real wide variant 1024/512 hidden units.
- No parameter counts; optimiser and learning rate not stated.

## Caveats

- One simulated phantom, one network shape, no seeds/error bars, no explicit parameter
  matching (the 2× real net is the only capacity control).
- The low-SNR result is the relevant one for noisy data and it is mixed: the complex net is
  not uniformly better once noise dominates.
