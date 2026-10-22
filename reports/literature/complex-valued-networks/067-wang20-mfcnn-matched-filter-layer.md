# Gravitational-wave signal recognition of LIGO data by deep learning

Wang, Wu, Cao, Cao, Zhang, 2020, Phys. Rev. D 101, 104003 ·
[arXiv:1909.13442](https://arxiv.org/abs/1909.13442)

**Evidence: B.** Peer-reviewed (PRD); 60 citations in 7.1 yr (Semantic Scholar, 2026-10-19).

## What they do

MFCNN is a CNN detector whose first layer is a **fixed matched-filter bank**: N_t = 35
whitened BBH templates (non-spinning, equal mass, M = 5 + 2i M☉, i = 0…34). The
frequency-domain inner product is rewritten as time-domain convolutions:

⟨d|h⟩(t) = 2 d̄(t) * h̄(−t),

where d̄ and h̄ are convolved with the whitening kernel (the inverse FT of S_n^{−1/2}). Both
whitening and matched filtering are therefore convolutions. The layer outputs the maximum
of each template's correlation, and a George & Huerta-style CNN follows.

The SNR definition they quote is ρ² = |⟨d|h⟩|² / ⟨h|h⟩, the modulus of the complex
correlation. The paper does not say how the layer realises the phase maximisation, for
example with two quadrature templates.

## What they find

- **Training:** O1 noise and SEOBNRE signals with total mass 10–150 M☉.
- **O1:** all three O1 events are found (GW150914 and GW151226 at confidence near 1,
  GW151012 above 0.5).
- **O2:** applied to O2 without retraining, all eight O2 events are found, including
  GW170817. The authors note that earlier CNNs found GW150914 but no other O1 event.
- **False triggers:** about 2000 (2242 after CAT3 vetoes, about one per 1.5 h). None match
  known subthreshold candidates or GRBs.
- **ROC and efficiency** are "comparable" to Gabbard et al. 2018 (file 68).
- "We need only tens of templates here instead of millions."

## Implementation details

- Input: 5 s at 4096 Hz from H1 and L1. 3220 training samples, with noise resampled every
  epoch.
- Curriculum on SNR, decreasing from 1 to 0.02 in their units. Adam at lr 3e-3, batch 16,
  Xavier init, BCE loss. About 6 h on four 1080 Ti.
- The template layer is fixed, not learned.

## Caveats

- There is no ablation of the matched-filter layer against an otherwise identical plain
  CNN. The gain is attributed to the layer by comparison with other papers' networks.
- The ROC numbers are only in the figures.
- Detection, not parameter estimation. The network is real-valued; the complex correlation
  lives inside the fixed filter.
- Details are from the ar5iv HTML.
