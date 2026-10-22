# Matching matched filtering with deep networks for gravitational-wave astronomy

Gabbard, Williams, Hayes, Messenger, 2018, Phys. Rev. Lett. 120, 141103 ·
[arXiv:1712.06041](https://arxiv.org/abs/1712.06041)

**Evidence: A.** Peer-reviewed (PRL); 176 citations in 8.8 yr (Semantic Scholar, 2026-10-19).

## What they do

A real-valued 1-D CNN classifies 1 s of whitened, simulated Advanced-LIGO strain (8192 Hz,
Gaussian design-PSD noise) as signal plus noise or noise only. Its ROC is compared with a
PyCBC matched-filter search: 8056 templates, 3% mismatch, and an SNR maximised over time,
**phase** and template.

- **Signals:** non-spinning IMRPhenomD BBH, 5–95 M☉.
- **Nuisance parameters:** phase and polarisation uniform on [0, 2π], isotropic sky, and a
  peak position jittered over 0.75–0.95 of the segment.
- **Training set:** 4 × 10⁵ series at fixed optimal SNR from 1 to 10.

## What they find

- The CNN "closely matches the sensitivity of matched-filtering" at optimal SNR 2, 4 and 6,
  over the false-alarm probabilities they could test (down to about 1e-4).
- It is slightly better than matched filtering at low SNR and high FAP, and marginally worse
  at high SNR and low FAP. They suggest more training data would close that gap.
- They argue that since matched filtering is optimal only for Gaussian noise, networks
  might beat it on real data. This is stated as a possibility, not shown.

## Implementation details

- 6 conv layers (filters 64, 32, 32, 16, 16, 16), 3 dense layers, ELU, 0.5 dropout, and a
  2-way softmax.
- Nadam at lr 2e-3.
- The network sees only real time-domain samples. Phase invariance has to be learned from
  the uniform-phase training draws.

## Caveats

- Detection is a scalar decision that is **invariant** to the signal phase. The network
  only has to learn a phase-maximised statistic, not to estimate the phase.
- Gaussian noise only, about 1000 unique waveforms, and a test set too small for FAP below
  1e-4.
- The architecture was tuned by trial and error.
- Details are from the ar5iv HTML.
