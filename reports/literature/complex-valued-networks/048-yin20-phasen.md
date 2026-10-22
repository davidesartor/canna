# PHASEN: A Phase-and-Harmonics-Aware Speech Enhancement Network

Yin et al., 2020, AAAI 2020 · [arXiv:1911.04697](https://arxiv.org/abs/1911.04697)

**Evidence: A.** Peer-reviewed (AAAI); 413 citations in 6.5 yr (Semantic Scholar,
2026-10-19).

## What they do

PHASEN is a real-valued network with **two streams**: an amplitude stream that predicts a
magnitude mask, and a phase stream that predicts phase. The streams exchange information at
every block.

- **Input**: the STFT as a T×F×2 tensor of (Re, Im); 25 ms Hann window, 10 ms hop, FFT 512.
- **Phase stream output**: 2 channels (Re, Im) normalised to unit amplitude per bin, i.e. a
  phasor rather than an angle.
- **Cross-stream link**: each stream's features are gated by the other's,
  x1 ∘ tanh(conv(x2)). The amplitude-to-phase direction is called "A2P" below.
- **Amplitude stream**: frequency transformation blocks to capture harmonics.
- **Loss**: 0.5 × MSE on the power-law compressed amplitude (|·|^0.3) + 0.5 × MSE on the
  compressed complex spectrum.

## What they find

- **Ablation**, SDR / PESQ:
  - single-stream cIRM: 15.08 / 2.87;
  - full PHASEN: 16.84 / 3.40.
- **Two streams vs one**: +0.85 dB SDR and +0.42 PESQ.
- **Without the amplitude-to-phase link**, phase estimation fails: the phase output collapses
  to Re ≈ 1, Im ≈ 0, i.e. the noisy phase is kept. Phase is learnable only when amplitude
  information informs it.
- **Output nonlinearity**: a ReLU after the last phase conv costs −5.52 dB, because it clips
  the signed RI output.
- **Voice Bank + DEMAND**: PESQ 2.99, CSIG 4.21, CBAK 3.55, COVL 3.62, SSNR 10.18.

## Implementation details

- The phase is represented as a unit-norm 2-vector per bin, never as an angle.
- Amplitude and phase are supervised jointly in the compressed domain.
- All layers are real.

## Caveats

- No complex-valued layers; this is about representation and architecture.
- Scores are mostly perceptual. The collapse of the phase stream without A2P is reported
  qualitatively.
