# A mathematical motivation for complex-valued convolutional networks

Tygert, Bruna, Chintala, LeCun, Piantino & Szlam, 2016, Neural Computation 28(5) ·
[arXiv:1503.03438](https://arxiv.org/abs/1503.03438)

**Evidence: A.** Peer-reviewed (Neural Computation); 109 citations in ~10 yr (Semantic
Scholar, 2026-10-19).

## What they do

A theory note: a complex convnet stage — convolve with complex filters, take the **modulus**,
then locally average — computes a data-driven **windowed (multiscale) spectrum**. With
windowed complex exponentials as filters, one stage is a short-time Fourier magnitude;
recursing on the zero-frequency channels gives a (multi)wavelet transform, recursing on all
channels a nonlinear wavelet-packet transform (the scattering-transform picture with learned
filters). The modulus is "the key nonlinearity": it removes the phase that a shift would
change, giving local translation invariance for (locally) stationary processes; the
averaging is a low-pass window that reduces variance and sets the scale.

## What they find

- The correspondence to wavelets/windowed spectra is exact for complex convnets with `|·|`,
  whereas real ReLU/max-pool nets have no such clean interpretation.
- CIFAR-10 numbers quoted from other work: scattering ≈ 18 % error; real convnet 13 % (11 %
  with response normalisation); complex convnet (Chintala et al.) 12 % with a net 8× the size
  of their earlier one. Real and complex nets are about equal when trained well.
- Remark: the absolute spectrum may be more robust than the power spectrum; both expected to
  perform similarly.

## Implementation details

Complex filters, `|·|` nonlinearity, local averaging (and decimation, which limits degrees of
freedom against overfitting). Suggests an alternative in which only per-filter scalings are
learned on a fixed multiwavelet bank.

## Caveats

- No new experiments; the CIFAR numbers are cited, not parameter-matched.
- The modulus is **phase-invariant** (`|e^{iα}z| = |z|`), i.e. this design deliberately
  *discards* phase. That suits classification of stationary textures/audio, where phase is a
  nuisance, and is the opposite of what is needed when the target is the phase itself.
