# CMGAN: Conformer-based Metric GAN for Speech Enhancement

Cao et al., 2022, Interspeech 2022 · [arXiv:2203.15149](https://arxiv.org/abs/2203.15149)

**Evidence: A.** Peer-reviewed (Interspeech); 182 citations in 4 yr (Semantic Scholar,
2026-10-19).

## What they do

CMGAN is a real-valued conformer network that enhances speech in the time-frequency domain.
It is notable for its input encoding, which mixes a compressed magnitude with compressed
Cartesian components.

- **Input**: three real channels, [|Y|^0.3, Yr_c, Yi_c]. Yr_c and Yi_c are the real and
  imaginary parts after power-law compression of the magnitude, keeping the original phase.
- **Encoder**: a dilated-DenseNet encoder, then two-stage conformers (time, then frequency).
- **Two decoders**:
  - a magnitude-mask decoder, applied to |Y|^0.3 and recombined with the noisy phase;
  - a complex decoder that predicts an RI residual, added to that estimate.
- **Loss**: 0.7 × MSE on compressed magnitude + 0.3 × MSE on compressed RI, plus a metric-GAN
  term (a PESQ-predicting discriminator) and a time-domain L1 term.
- **Size**: 1.83M parameters.

## What they find

- **Voice Bank + DEMAND**: PESQ 3.41, CSIG 4.63, CBAK 3.94, COVL 4.12, SSNR 11.10, STOI 0.96.
- **Ablations**, PESQ / SSNR:

  | Variant | PESQ | SSNR |
  |---|---|---|
  | Full model | 3.41 | 11.10 |
  | Magnitude-only path | 3.23 | 9.82 |
  | Complex (RI)-only path | 3.35 | 9.19 |
  | Without time loss | 3.45 | 9.71 |
  | Without discriminator | 3.24 | — |

- **Takeaway**: magnitude alone and RI alone are each worse than the combination; RI-only
  beats magnitude-only on PESQ but not on SSNR.

## Implementation details

- **Compression**: power-law with exponent 0.3 on the magnitude, applied before forming the RI
  channels. This reduces dynamic range while keeping the phase exact.
- All layers are real-valued; RI is just extra channels.
- The decoder outputs are combined in the Cartesian domain.

## Caveats

- No complex-valued layers are tested. This is evidence about input/target representation,
  not about complex arithmetic.
- Single benchmark (Voice Bank + DEMAND), with perceptual metrics only.
- The "without time loss" ablation raises PESQ but lowers SSNR, so the metrics disagree.
