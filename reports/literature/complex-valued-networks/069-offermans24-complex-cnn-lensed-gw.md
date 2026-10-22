# Using time series to identify strongly-lensed gravitational waves with deep learning

Offermans, Li, 2024, preprint · [arXiv:2411.12453](https://arxiv.org/abs/2411.12453)

**Evidence: C.** Not peer-reviewed as far as the arXiv record shows; 1 citation in 1.9 yr
(Semantic Scholar, 2026-10-19).

## What they do

This is the only GW paper we found that uses a **complex-valued network** (complex weights)
on strain data. The task is to classify whether two events are strongly-lensed copies of
one signal. They argue that time-frequency images, the usual input, throw away phase, and
that lensing imprints a phase shift between images. So they work on the raw time series
with complex layers.

- **Input:** 6 × 1024 (2 events × 3 detectors, 0.5 s at 2048 Hz), whitened with O3 PSDs
  and high-passed at 20 Hz.
- **Making it complex:** the imaginary part is set to zero. There is no analytic signal and
  no Fourier transform.
- **Complex part:** 4 complex conv layers (kernel 3, starting at 64 channels and doubling),
  each followed by ℂReLU (ReLU on real and imaginary parts separately, after Trabelsi et
  al. 2018) and complex average pooling.
- **Real part:** the real and imaginary parts are flattened, concatenated with the time
  delay Δt, and passed to 4 real dense layers ending in a sigmoid.
- **Init:** Rayleigh-distributed magnitudes at He/Glorot variance, with uniform phases.
  PyTorch.

## What they find

- Better than the time-frequency baseline (G+21, a DenseNet201 on spectrograms) at all
  false-positive rates, with TPR up to about 5× higher at FPR 1e-4.
- Robust to waveform-model choice, lensing phase shifts and merger-time errors up to 20 ms.
- Inference: 8.4 ms per batch of 500 on a GTX 750.

## Implementation details

- Training uses 10⁵ waveforms, SNR uniform on [7, 30], and fresh noise each epoch.
- Optimiser, learning rate and epochs are not given in the text.

## Caveats

- **There is no real-valued ablation.** The comparison is against a different
  architecture, a different input representation (spectrogram vs time series) and a
  different dataset. The gain cannot be attributed to complex weights rather than to using
  the time series.
- With a zero imaginary input and split ℂReLU, the first complex conv acts on a real signal.
  The complex structure is mostly extra channels.
- Gaussian noise only, no lens population.
- Details are from the arXiv HTML.
