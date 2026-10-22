# Impact of PolSAR pre-processing and balancing methods on complex-valued neural networks segmentation tasks

Barrachina, Ren, Vieillard, Morisseau & Ovarlez, 2023, IEEE Open Journal of Signal Processing ·
[arXiv:2210.17419](https://arxiv.org/abs/2210.17419)

**Evidence: B.** Peer-reviewed (IEEE OJSP, per DBLP/Semantic Scholar); 4 citations in ~3 yr
(Semantic Scholar, 2026-10-19).

## What they do

Semantic segmentation of polarimetric SAR (natively complex, X-band Bretigny scene, 4
classes) with complex-valued MLP, CNN and fully-convolutional (FCNN, U-Net-like) networks,
each against a **real-valued equivalent** fed (Re, Im) as separate channels. Also studies
input representation (Pauli vector vs coherency matrix) and dataset splitting/balancing.

## What they find

Overall accuracy (OA, %), CV vs RV, medians over repeated runs:

| Model / input | Standard split | Dataset split |
|---|---|---|
| CNN / coherency | 95.78 vs 94.43 | 93.92 vs 93.31 |
| CNN / Pauli | 95.40 vs 94.78 | 93.03 vs 92.52 |
| MLP / coherency | 95.09 vs 95.13 | 93.40 vs 92.34 |
| MLP / Pauli | 88.55 vs 87.77 | 81.47 vs 82.29 |
| FCNN | 99.83 vs 99.69 | 93.62 vs 92.63 |

- Complex wins in most cells but by **< 1.5 OA points**, and loses in two (MLP/coherency on
  the standard split; MLP/Pauli on the dataset split). Average accuracy shows larger swings
  in both directions (CNN/coherency standard split AA 89.72 vs 85.82; FCNN dataset split AA
  75.31 vs 76.20). Several confidence intervals overlap.
- The split/balancing choice (standard vs disjoint-region split) moves accuracy far more
  than the CV/RV choice; the standard random-patch split inflates scores through spatial
  leakage.

## Implementation details

- **Parameter matching**: the real-valued equivalents are built to have the **same number of
  trainable real parameters** as the complex models; for MLPs the hidden widths are chosen to
  keep the complex net's layer aspect ratio. Complex models: MLP hidden 96/180 units, CNN 6/12
  kernels.
- Adam, He-style init (complex variant for the CVNN), built with the authors' `cvnn` library.
  Split-type activations (complex ReLU variants). No precision discussion.

## Caveats

- Single scene, small models, small gaps; the per-cell numbers above were read from the arXiv
  HTML tables and some CIs overlap, so most differences are not individually significant.
- Venue is per DBLP; the arXiv version is the one read.
