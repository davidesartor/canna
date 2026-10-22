# Learning Complex Spectral Mapping With Gated Convolutional Recurrent Networks for Monaural Speech Enhancement

Tan & Wang, 2020, IEEE/ACM TASLP 28 ·
[DOI:10.1109/TASLP.2019.2955276](https://doi.org/10.1109/TASLP.2019.2955276)

**Evidence: A.** Peer-reviewed (IEEE/ACM TASLP); 381 citations in 7 yr (Semantic Scholar,
2026-10-19; OpenAlex 349).

## What they do

GCRN is a **real-valued** network that maps the noisy complex spectrum, fed as stacked (Re, Im)
channels, directly to the clean (Re, Im) spectrum. This is "target complex spectrum", TCS:
the network regresses the spectrum rather than a mask.

- **STFT**: 16 kHz, 20 ms Hamming window, 50% overlap, 320-point FFT (161 bins).
- **Encoder**: 5 Conv-GLU blocks (kernel 1×3, stride (1, 2), channels 16/32/64/128/256).
- **Bottleneck**: two grouped LSTMs of 1024 units (G = 1/2/4/8).
- **Decoders**: two parallel decoders, one for Re and one for Im, each 5 Deconv-GLU blocks with
  skip connections. BN + ELU throughout; causal.
- **Training**: MSE on RI, AMSGrad lr 0.001, batch 4.
- **Data**: WSJ0 SI-84 with 10k noises (320k mixtures at −5..0 dB). Tests on babble and
  cafeteria noise at −5/0/5 dB.
- **Comparisons**: target magnitude spectrum (TMS) and two complex masks, the cIRM and a
  signal-approximation cRM (cRM-SA).

## What they find

- **Average STOI (%) / PESQ** at −5 / 0 / 5 dB:

  | System | −5 dB | 0 dB | 5 dB |
  |---|---|---|---|
  | Unprocessed | 57.84 / 1.50 | 69.80 / 1.80 | 81.06 / 2.12 |
  | LSTM + TMS | 74.40 / 1.94 | 85.15 / 2.40 | 91.12 / 2.78 |
  | LSTM + TCS | 78.11 / 2.00 | 88.15 / 2.50 | 92.69 / 2.87 |
  | CRN + TMS | 75.80 / 2.00 | 86.27 / 2.46 | 91.99 / 2.85 |
  | GCRN + cIRM | 74.75 / 1.94 | 85.98 / 2.44 | 92.16 / 2.86 |
  | GCRN + cRM-SA | 78.43 / 2.03 | 88.82 / 2.56 | 93.72 / 2.94 |
  | CNN + TCS (Fu et al.) | 67.61 / 1.66 | 81.58 / 2.14 | 89.26 / 2.51 |
  | GCRN + TCS, G = 1 | 80.71 / 2.11 | 89.96 / 2.62 | 94.18 / 3.00 |
  | GCRN + TCS, G = 2 | 80.75 / 2.13 | 90.12 / 2.65 | 94.39 / 3.03 |

- **Target**: RI spectral mapping beats magnitude-only mapping by about 3–6 STOI points. The
  margin is **largest at −5 dB**. RI mapping also beats cIRM masking.
- **Phase distance** (babble), noisy / cIRM / TCS:

  | SNR | Noisy | cIRM | TCS |
  |---|---|---|---|
  | −5 dB | 30.896° | 29.516° | 24.151° |
  | 0 dB | 21.991° | 20.342° | 17.271° |
  | 5 dB | 14.885° | 13.842° | 12.957° |

  Cafeteria at −5 dB: 38.644° / 36.503° / 30.398°. Phase error, and the gain from TCS, both
  grow as SNR falls.
- **Value of the estimated phase alone**: combining the TCS-estimated phase with an enhanced
  magnitude at −5 dB gives 78.25% / 2.15, against 75.80% / 2.00 with the noisy phase and
  80.30% / 2.35 with the clean phase.
- **Authors' framing**: "complex spectral mapping provides an effective phase estimation".
  Phase spectra "lack spectrotemporal structure", whereas RI spectra are learnable.

## Implementation details

- No magnitude compression is applied to the RI input or target.
- Separate Re and Im decoders share one encoder.
- GLU gating is real-valued.

## Caveats

- This is not a complex-valued network. It is evidence for **Cartesian RI targets** in real
  networks.
- The numbers come from the PMC author manuscript, read via a fetched summary, and were not
  cross-checked against the typeset tables.
