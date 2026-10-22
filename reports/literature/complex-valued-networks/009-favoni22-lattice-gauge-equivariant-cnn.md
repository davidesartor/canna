# Lattice Gauge Equivariant Convolutional Neural Networks

Favoni, Ipp, Müller & Schuh, 2022, Physical Review Letters 128, 032003 ·
[arXiv:2012.12901](https://arxiv.org/abs/2012.12901)

**Evidence: B.** Peer-reviewed (PRL); 80 citations in 5.8 yr (Semantic Scholar, 2026-10-19).

## What they do

L-CNNs: convolutional networks on SU(N) lattice gauge fields that are exactly equivariant to
local gauge transformations (and to translations). Building blocks:

- **Plaq:** compute plaquettes (smallest Wilson loops) as gauge-covariant input features.
- **L-Conv:** parallel-transport neighbouring features along the links before a weighted sum,
  so the convolution stays covariant.
- **L-Bilin:** bilinear products of covariant features (multiplying matrices), the step that
  grows longer Wilson loops layer by layer.
- **L-Act:** activations that rescale a covariant matrix by a function of a gauge invariant
  (e.g. its trace).
- **Poly:** Polyakov loops; a final trace makes the output gauge-invariant.

They argue that stacking L-Conv + L-Bilin can generate Wilson loops of any shape, so the class
can in principle approximate any gauge-covariant function.

## What they find

- SU(2) data from Metropolis Monte Carlo; regression of traced Wilson loops (1+1D) and
  topological charge density (3+1D). Trained on small lattices (8·8 for 1+1D, 10⁴ training
  samples), tested on lattices up to 64·64.
- L-CNNs of ≈100 to ≈40,000 parameters vs **2680 trained baseline CNNs** (≈300 to ≈100,000
  parameters, many widths, depths and activations), the baselines also given precomputed
  plaquettes.
- Baselines "quickly deteriorate with the growing size of the Wilson loop"; for 4 × 4 loops
  "the baseline model collapses and only predicts the average value of the training data". Only
  1 × 1 loops are learned adequately by the baselines. L-CNNs fit all loop sizes and transfer
  to larger lattices without retraining (scatter plots, MSE in figure insets).
- Feeding plaquettes to the baselines improved them by "many orders of magnitude" for 1 × 1
  loops (supplement), i.e. a non-equivariant net fails even to assemble the invariant product.

## Implementation details

- Gauge links are complex 2 × 2 matrices; L-CNN weights are complex. Baseline inputs split
  links into real and imaginary channels. PyTorch, MSE loss.

## Caveats

- Toy regression targets whose exact form is known; a Wilson loop is a product of many
  link phases, so the target is maximally sensitive to every input phase — a
  "no partial credit" coherent quantity that unconstrained nets fit only by predicting the
  mean.
- Results mostly as figures; the advantage is shown for exact local symmetries that the
  baselines must learn from 10⁴ examples.
