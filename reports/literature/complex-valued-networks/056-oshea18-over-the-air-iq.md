# Over-the-Air Deep Learning Based Radio Signal Classification

O'Shea, Roy & Clancy, 2018, IEEE Journal of Selected Topics in Signal Processing ·
[arXiv:1712.04578](https://arxiv.org/abs/1712.04578) ·
[DOI:10.1109/JSTSP.2018.2797022](https://doi.org/10.1109/JSTSP.2018.2797022)

**Evidence: A.** Peer-reviewed (IEEE JSTSP); 1516 citations in 8.5 yr (Semantic Scholar,
2026-10-19).

## What they do

This is the reference study for modulation classification straight from raw complex baseband.
The input is **I/Q as 2 real channels × 1024 samples**, normalised to unit variance, with no
expert features. The authors state: "Real valued networks are used, as complex valued
auto-differentiation is not yet mature enough for practical use."

- **Models**:
  - a VGG-style 1D CNN: 7 conv + maxpool layers of 64 channels, then fully connected SELU
    128-128 and a 24-way softmax (257,099 parameters);
  - a ResNet: 6 residual stacks of 32 channels, SELU fully connected layers and alpha dropout
    (236,344 parameters).
- **Baseline**: XGBoost on 28 higher-order moment and cumulant features.
- **Data**:
  - 24 modulation classes ("Difficult" set) and an 11-class "Normal" set;
  - Es/N0 from −20 to +30 dB;
  - simulated carrier-frequency offset, sample-rate offset and Rayleigh fading;
  - over-the-air captures with USRP B210 radios at 900 MHz.

## What they find

- **Accuracy at high SNR, 11 classes**: ResNet 99.8%, VGG 98.3%, feature baseline 94.6%. The
  ResNet has about 5 dB better sensitivity.
- **24 classes, AWGN**: the ResNet is 2–6 dB better than the other models.
- **Under local-oscillator impairment**: nearly 6 dB and +20% accuracy at high SNR over the
  baseline.
- **Across impairments, high SNR**: the ResNet ranges from about 80% (best case) to 59%
  (worst); the baseline from 61% to 45%.
- **Model size and depth**: the L = 6 ResNet (121 layers, 229k parameters) beats shallower,
  larger models (L = 0: 2.1M parameters).
- **Data scaling**: near-random accuracy until 4–8k examples. About 95% at high SNR with
  1–2M examples, with diminishing returns past 1M.
- **Window length**: about +3% accuracy per doubling of the window, up to 512–1024 samples.
- **Over the air**:
  - training and testing on OTA data gives 95.6% test accuracy (1.44M examples, about 10 dB);
  - synthetic → OTA transfer without updates gives 64–80%, and fine-tuning the last layers
    adds about 10%;
  - a model trained with clock-drift σ_clk = 0.0001 scores 94% on synthetic and 87% on OTA
    data.
- **LO offset as a regulariser**: adding LO offset during training improves the ResNet.

## Implementation details

- The I/Q input is treated as a 2-channel 1D signal. The first convolution mixes I and Q
  freely, with no complex constraint.
- Normalisation is to unit average power; there is no log or polar transform of the input.
- SELU activations and alpha dropout are used.

## Caveats

- **No complex-vs-real comparison.** It is included as the baseline real-on-I/Q design that
  later complex-valued modulation classifiers (files 54, 55) modify.
- The OTA data come from a single hardware setup.
