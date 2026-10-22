# Phase-Aware Speech Enhancement with Deep Complex U-Net

Choi et al., 2019, ICLR 2019 · [arXiv:1903.03107](https://arxiv.org/abs/1903.03107)

**Evidence: A.** Peer-reviewed (ICLR); 407 citations in 7.5 yr (Semantic Scholar,
2026-10-19).

## What they do

DCUnet is a U-Net built from complex convolutions (Trabelsi-style). It estimates a complex
ratio mask on the STFT of noisy speech and is trained with a time-domain loss.

- **Data**: Voice Bank + DEMAND.
- **STFT**: 64 ms window, 16 ms hop.
- **Mask**: a polar, bounded cRM. The magnitude is tanh(|O|) and the phase is O/|O| of the
  network output O. The motivation is that unbounded masks are hard to train and bounded
  ones generalise better.
- **Loss**: weighted SDR (wSDR), computed after the inverse STFT. It weights the speech and the
  noise estimates by their energies.
- **Ablation**: real vs complex arithmetic, crossed with real (magnitude) vs complex masks.

## What they find

- **Results**, CSIG / CBAK / COVL / PESQ / SSNR:
  - DCUnet-20: 4.24 / 4.00 / 3.69 / 3.13 / 15.95.
  - Large DCUnet: 4.34 / 4.10 / 3.81 / 3.24 / 16.85.
- **Table 3, matched parameter counts**. PESQ for three settings: complex net + complex mask
  (cRMCn), real net + complex mask (cRMRn), and real net + real magnitude mask (RMRn):

  | Depth | cRMCn | cRMRn | RMRn |
  |---|---|---|---|
  | 20-layer | 3.13 | 3.07 | 2.74 |
  | 16-layer | 2.93 | 2.91 | 2.66 |
  | 10-layer | 2.72 | 2.72 | 2.51 |

  SSNR at 20 layers is 15.95 / 15.54 / 9.90.
- **What drives the gain**: almost all of it comes from **estimating phase at all**
  (cRM vs magnitude mask: +0.21–0.33 PESQ; +5.6 dB SSNR at 20 layers). Complex vs real
  arithmetic at equal parameters adds 0–0.06 PESQ.
- **Phase improvement grows at low SNR** (Table 6): 6.7° at 2.5 dB input SNR vs 1.7° at
  17.5 dB.
- **Loss**: with a spectrogram MSE loss the learned cRM's imaginary part collapses to about 0,
  i.e. the network ignores phase. The time-domain wSDR loss avoids this.

## Implementation details

- Complex convolution is implemented with four real convolutions. The paper uses complex BN
  and a leaky CReLU.
- The bounded polar mask applies tanh to the magnitude only and leaves the output phase
  unconstrained.
- The loss is wSDR in the time domain, so phase errors are penalised through the waveform.

## Caveats

- **The arXiv entry is withdrawn.** v2 (2019-04-02) reads "Significant error was found in data
  processing step, therefore will be retracted". All numbers above are from v1 / the ICLR
  OpenReview version. The paper remains widely cited and reimplemented, but the absolute scores
  should not be trusted.
- Real and complex models are matched on parameters, not on compute.
