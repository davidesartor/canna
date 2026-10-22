# Deep complex networks

Trabelsi, Bilaniuk, Zhang, Serdyuk, Subramanian, Santos, Mehri, Rostamzadeh, Bengio & Pal,
2018, ICLR · [arXiv:1705.09792](https://arxiv.org/abs/1705.09792)

**Evidence: A.** Peer-reviewed (ICLR 2018); 1101 citations in ~8 yr (Semantic Scholar,
2026-10-19).

## What they do

The reference recipe for deep complex networks: complex convolution, complex batch
normalisation, complex weight initialisation and a comparison of complex activations, in
residual CNNs and a convolutional LSTM. Tested on vision (CIFAR-10/100, SVHN), music
transcription from the audio spectrum (MusicNet) and speech spectrum prediction (TIMIT).

## What they find

- **Vision** (≈ 1.7 M params each, complex vs real best error): CIFAR-10 5.59 vs 5.42 %,
  CIFAR-100 26.36 vs 27.22 %, SVHN 3.62 vs 3.42 %. A wash: real images carry no phase.
- **MusicNet** (multi-label note transcription, 84 outputs; input a 4096-sample window at
  11 kHz or its FFT): shallow nets tie (AP 66.0 complex vs 66.1 real); **deep VGG-like nets
  on the complex spectrum: 72.9 vs 69.6 AP**, with the complex net *smaller* (8.8 M vs 10.0 M
  params). The real net takes the spectrum as separate Re and Im channels, so both see the
  same information. Claimed SOTA at the time.
- **TIMIT** (predict next log-magnitude STFT frame from the full complex STFT, Re/Im as
  channels for both): ConvLSTM 12.18 vs complex ConvLSTM 11.90 test MSE at ≈ 88 k params
  each (84 real vs 60 complex maps).
- **Activations**: CReLU (separate ReLU on Re and Im) was best and always converged;
  modReLU and zReLU "failed to converge or performed substantially worse" on vision.

## Implementation details

- **Complex conv**: `W*h = (A*x − B*y) + i(B*x + A*y)` for `W = A+iB`, `h = x+iy`; implemented
  with real convs on stacked Re/Im feature maps.
- **Complex BN**: centre, then whiten each complex unit by the inverse square root of its
  2×2 (Re, Im) covariance; learnable 2×2 scale γ (init `γ_rr = γ_ii = 1/√2`, `γ_ri = 0`) and
  complex shift β (init 0); running stats momentum 0.9. Not phase-equivariant (γ is a general
  2×2 matrix and the stats are per batch).
- **Init**: magnitude Rayleigh with mode σ, phase uniform on [−π, π], so `Var(W) = 2σ²`; Glorot
  `2/(n_in+n_out)` or He `2/n_in`; also a (semi-)unitary variant rescaled to that variance.
- **Activations**: modReLU `ReLU(|z|+b) z/|z|`; CReLU `ReLU(Re z) + i ReLU(Im z)`; zReLU
  passes z only if `arg z ∈ [0, π/2]`.
- For real images the imaginary input is **learned** by a small real residual block; this
  beat setting it to zero.
- Optimisers: SGD + Nesterov 0.9, grad-norm clip 1 (vision); Adam 1e-3 (MusicNet),
  1e-4 (TIMIT). Numeric precision not discussed.

## Caveats

- "Matched" means roughly equal totals (1.7 M vision; 8.8 vs 10.0 M MusicNet, the complex
  net *smaller*; 88 k TIMIT), no formal rule. The modReLU vision models had 2.5 M params.
- The one clear win (MusicNet +3.3 AP) is a single architecture/seed comparison on complex
  spectral input; the shallow tie shows the gain is not automatic.
- The winning activation, CReLU, is not phase-equivariant; the equivariant modReLU trained
  worst. Complex BN whitening (2×2 inverse square root per unit) is costly.
