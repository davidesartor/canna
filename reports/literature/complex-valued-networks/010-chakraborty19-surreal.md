# SurReal: Complex-Valued Learning as Principled Transformations on a Scaling and Rotation Manifold

Chakraborty, Xing & Yu, 2020 (2022 issue), IEEE TNNLS 33(3):940–951 ·
[arXiv:1910.11334](https://arxiv.org/abs/1910.11334); workshop version Chakraborty, Wang & Yu,
"Sur-Real: Fréchet Mean and Distance Transform for Complex-Valued Deep Learning", CVPR 2019
workshops (Best Paper award) · [arXiv:1906.10048](https://arxiv.org/abs/1906.10048)

**Evidence: B.** Peer-reviewed (IEEE TNNLS); 28 citations in 7.0 yr (OpenAlex, 2026-10-19;
Semantic Scholar rate-limited).

## What they do

Treat each complex number as a point on the Riemannian manifold ℝ⁺ × SO(2) (log-magnitude ×
phase) and build layers equivariant to complex scaling `z → s z` (gain and phase), the
predecessor of the co-domain-symmetry work (file 07):

- **wFM convolution:** the weighted Fréchet mean of the inputs in the receptive field on
  ℝ⁺ × SO(2) (weights convex). Equivariant to scaling by construction; computed iteratively,
  no closed form.
- **tReLU:** on the manifold, `r → max(r, 1)`, `θ → max(θ, 0)`.
- **Distance-transform layer:** distance of each feature to the channel wFM, making the
  representation invariant to complex scaling before a real-valued head.
- G-transport (a per-channel learned complex multiplier) and concatenation residuals in the
  TNNLS version.

## What they find

- MSTAR SAR target recognition (L/S splits): SurReal 99.1 / 97.7, ResNet50 99.1 / 97.4,
  DCN (Trabelsi) 98.9 / 93.3, SurReal with residuals 99.2 / 98.4. Params: ResNet50 23.5M,
  DCN 154K, SurReal 67K (21K with a tensor-ring factorisation).
- With 5% of the training data: SurReal ≈90% vs DCN ≈60% and ResNet ≈45% (read from figure).
- RadioML modulation classification at SNR 10 dB: SurReal 76.1% with 20K params vs O'Shea
  CNN 72.7% (2.8M) and DCN 76.3% (35K).
- **At low SNR (−15 to −5 dB) SurReal underperforms the baselines** on RadioML (figure).
- Workshop abstract: MSTAR 94% → 98% at 8% of the baseline size; RadioML comparable accuracy
  at 10% of the size.

## Implementation details

- Polar processing throughout: magnitudes in log space, phases as angles. Robustness to
  complex scaling shown under random input scaling.

## Caveats

- Main gain is parameter efficiency at equal accuracy, not higher accuracy at equal size.
- The polar representation weakens exactly where noise dominates: phase becomes ill-defined as
  magnitude falls to noise level, and the low-SNR RadioML regime is where SurReal loses.
- On CIFAR/SVHN (via colour encoding) it is the weakest model in file 07's comparison.
