# Complex-Valued Convolutional Neural Networks for Enhanced Radar Signal Denoising and Interference Mitigation

Fuchs et al., 2021, IEEE RadarConf 2021 ·
[arXiv:2105.00929](https://arxiv.org/abs/2105.00929) ·
[DOI:10.1109/RadarConf2147009.2021.9455296](https://doi.org/10.1109/RadarConf2147009.2021.9455296)

**Evidence: B.** Peer-reviewed (IEEE RadarConf); 44 citations in 5.5 yr (Semantic Scholar,
2026-10-19; OpenAlex 52).

## What they do

The task is denoising and interference mitigation on automotive FMCW range-Doppler (RD) maps.
The aim is to recover both magnitude and phase of target peaks, since phase feeds later
angle estimation.

- **Data**: real measured RD maps with simulated interference added; 2500 training maps of
  96×96, scaled to zero mean and unit variance.
- **Input**: (Re, Im) as 2 channels for the real nets; one complex channel for the complex
  nets.
- **Complex net (CVCNN)**: 3 layers with 3×3 kernels.
  1. Complex conv + CReLU.
  2. Complex conv + complex batch norm (Trabelsi 2×2 whitening with Tikhonov ε) + CReLU.
  3. Complex conv.
- **Real baseline**: the same structure with real channels.
- **Sweep**: 24 architectures per family.
  - Complex: C1 ∈ {1..32}, C2 ∈ {2..16}, C3 = 1.
  - Real: C1 ∈ {1..32}, C2 ∈ {4..32}, C3 = 2.
- **Comparison**: the two families are compared per parameter count and per FLOP.
- **Training**: MSE on Re and Im, Adam 5e-3, batch 8, 100 epochs.
- **Metrics**: detection F1, error vector magnitude (EVM) and phase MSE at peaks (PPMSE,
  rad²).

## What they find

- **Parameter efficiency**: CVCNNs beat real CNNs at every parameter count. Complex models
  under 500 parameters match the largest real models. Per FLOP the advantage is smaller, but
  complex still wins for most architectures.
- **Data efficiency** (Table I; fractions 20/40/60/80/100% of the training set). Two complex
  nets are compared with R 16-8-2: C 8-8-1 is parameter-matched to it, C 8-4-1 is
  FLOP-matched.

  | Model | F1 | EVM | PPMSE |
  |---|---|---|---|
  | C 8-8-1 | .895/.898/.896/.898/.897 | .163/.136/.124/.117/.115 | .017/.014/.013/.013/.013 |
  | C 8-4-1 | .897/.896/.897/.896/.897 | .163/.143/.133/.127/.122 | .018/.016/.015/.014/.014 |
  | R 16-8-2 | .872/.881/.887/.891/.885 | .258/.262/.209/.184/.146 | .049/.054/.035/.022/.018 |

  Complex reaches near-top results with about 10% of the data; real needs ≥50%. The phase
  error gap is about 3× at 20% data and shrinks to about 1.4× at 100%.
- **Best models** (Table II; selected by best EVM), F1 / EVM / PPMSE / parameters / MFLOP:
  - R-16-16-2: 0.896 / 0.123 / 0.015 / 2946 / 53.24.
  - C-32-16-1: 0.902 / 0.098 / 0.011 / 10242 / 374.27.
  - The complex model is about 3.5× larger and uses 7× the FLOPs, so this comparison is not
    matched.
  - Classical baselines: zeroing 0.856 / 0.124 / 0.069; RFmin 0.875 / 0.145 / 0.054; IMAT
    0.860 / 0.119 / 0.065.
  - Reference F1: no mitigation 0.837; target 0.909.
- **Where the phase gain is largest**: for weak interference.

## Implementation details

- Inputs are standardised RD maps, with no log or compression of magnitude.
- The output is the complex RD map: a single complex channel for complex nets, 2 real channels
  for real nets.
- Complex BN follows Trabelsi et al. 2018: whitening by the inverse square root of the 2×2
  (Re, Im) covariance.
- The activation is CReLU (split).

## Caveats

- Networks are tiny (≤10k parameters) and use a single dataset. The large-model regime, where
  the real network could learn the complex structure, is not tested.
- Metrics are evaluated at target peaks only.
- The best-vs-best table compares models of different size.
