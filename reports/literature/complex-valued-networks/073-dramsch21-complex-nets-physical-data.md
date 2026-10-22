# Complex-valued neural networks for machine learning on non-stationary physical data

Dramsch, Lüthje, Christensen, 2021, Computers & Geosciences 146, 104643 ·
[arXiv:1905.12321](https://arxiv.org/abs/1905.12321)

**Evidence: B.** Peer-reviewed (Computers & Geosciences); 53 citations in 7.4 yr (Semantic
Scholar, 2026-10-19).

## What they do

A controlled comparison of real-valued and complex-valued convolutional autoencoders on
physical waveform data: 64 × 64 patches of the F3 North Sea seismic volume.

- **Complex input:** the analytic signal of each trace (Hilbert transform), so it carries
  the instantaneous phase explicitly.
- **Layers:** complex convolutions built from four real convolutions,
  (M_R*K_R − M_I*K_I) + i(M_R*K_I + M_I*K_R), and complex batch norm with 2 × 2 covariance
  whitening, both after Trabelsi et al.
- **Pairs:** "small" and "large" networks with about half the real parameter count on the
  complex side.
- **Training:** 7 seeds each, Adam at lr 1e-3, MSE loss.

## What they find

Validation results (Table 2, MSE ×1e-2):

| network | params | MSE | MAE (×1e-2) |
|---|---|---|---|
| complex small | 100k | 0.484 ± 0.013 | 4.70 |
| real small | 198k | 0.436 ± 0.006 | 4.50 |
| complex large | 397k | 0.227 ± 0.003 | 3.25 |
| real large | 791k | 0.196 ± 0.002 | 3.05 |

- **Full test data** (Table 3, RMS): complex small 0.155 vs real small 0.158; complex large
  0.151 vs real large 0.147.
- **Qualitatively**, complex networks reconstruct faults and discontinuities with less
  smearing and fewer dipping artefacts.
- **The abstract overstates it.** It claims smaller complex nets beat larger real ones, but
  the tables do not support that cleanly. The large real network has the best numbers
  overall.

## Implementation details

- Keras/TensorFlow reimplementation of Trabelsi et al.'s complex layers.
- The complex conv costs 4 real convolutions per layer, and the paper notes a higher memory
  footprint.

## Caveats

- **The comparison is confounded.** The complex nets get the analytic trace, with explicit
  phase, while the real nets get the raw trace. Input and architecture both change.
- One dataset, one reconstruction task, mostly visual evaluation.
- The activation function is not stated.
- Details are from the ar5iv HTML.
