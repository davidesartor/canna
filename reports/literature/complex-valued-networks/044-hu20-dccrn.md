# DCCRN: Deep Complex Convolution Recurrent Network for Phase-Aware Speech Enhancement

Hu et al., 2020, Interspeech 2020 · [arXiv:2008.00264](https://arxiv.org/abs/2008.00264)

**Evidence: A.** Peer-reviewed (Interspeech); 838 citations in 6 yr (Semantic Scholar,
2026-10-19).

## What they do

DCCRN is a convolution-recurrent encoder/decoder for speech enhancement with complex Conv2d,
complex batch norm and real PReLU, and an optional complex LSTM in the bottleneck. A complex
LSTM is two real LSTMs combined like a complex multiply.

- **Loss**: SI-SNR on the waveform. The STFT and iSTFT are implemented as fixed convolutions,
  so the loss is end-to-end.
- **STFT**: 25 ms window, 6.25 ms hop, FFT 512.
- **Encoder**: channels {32, 64, 128, 128, 256, 256}, kernel (5, 2).
- **Three mask types**:
  - R: separate Re and Im masks.
  - C: a cRM applied as a complex multiply.
  - E: a polar mask, with a tanh-bounded magnitude and a phase taken from the angle.
- **CL variant**: a complex LSTM of 128 + 128 units.
- **Size**: all DCCRNs have 3.7M parameters.
- **Baselines**:
  - CRN: real, with RI as 2 channels; 6.1M parameters, not matched.
  - LSTM: magnitude only; 9.6M parameters.
  - DCUNET-16: 3.6M parameters.

## What they find

- **WSJ0 + MUSAN**, average PESQ:

  | Model | Average PESQ |
  |---|---|
  | Noisy | 2.518 |
  | LSTM | 3.326 |
  | CRN | 3.329 |
  | DCCRN-R | 3.424 |
  | DCCRN-C | 3.409 |
  | DCCRN-E | 3.433 |
  | DCCRN-CL | 3.498 |
  | DCUNET | 3.500 |

  At 0 dB: CRN 2.850, DCCRN-E 2.859, DCCRN-CL 2.972.
- **DNS Challenge 2020** (Table 2), PESQ without / with reverb:
  - DCCRN-E 3.266 / 3.077.
  - DCCRN-CL 3.262 / 3.101.
  - DCUNET 3.223 / 2.796.

  DCCRN-E took 1st place in the real-time track (MOS 3.42).
- **Compute**: DCUNET costs about 6× the compute of DCCRN-CL for similar quality.
- **Size of the gain**: the gain over the real RI baseline (CRN) is about +0.1 PESQ, with fewer
  parameters. However, the architectures differ and are not parameter-matched, so the gain
  cannot be attributed to complex arithmetic alone.

## Implementation details

- The nonlinearity is real PReLU applied to Re and Im separately.
- Complex BN is used in the encoder and decoder.
- The bounded polar mask (E) is the variant submitted to the DNS Challenge.
- The SI-SNR loss is on the waveform, so phase is supervised implicitly.

## Caveats

- There is no parameter-matched real twin. Wu et al. 2023 (file 40) later built one and found
  the real DCCRN equal (PESQ 1.80 vs 1.78 at −5 dB) at about one third of the MACs.
- PESQ-centric evaluation; no direct phase metric is reported.
