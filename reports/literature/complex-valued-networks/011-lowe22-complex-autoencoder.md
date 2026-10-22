# Complex-Valued Autoencoders for Object Discovery

Löwe, Lippe, Rudolph & Welling, 2022, Transactions on Machine Learning Research (11/2022) ·
[arXiv:2204.02075](https://arxiv.org/abs/2204.02075)

**Evidence: B.** Peer-reviewed (TMLR); 9 citations in 4.5 yr (OpenAlex, arXiv record only,
likely an undercount; Semantic Scholar rate-limited, 2026-10-19).

## What they do

The Complex AutoEncoder (CAE): a convolutional autoencoder with complex activations, trained
only on reconstruction of real grayscale images, whose output **phases** group pixels by object
(temporal-correlation / synchrony hypothesis). Input image becomes magnitudes with phase 0.
Each layer, with real weights `w` applied to real and imaginary parts separately:

- `ψ = f_w(Re z) + i f_w(Im z)` (complex-linear with real weights);
- magnitude `m_ψ = |ψ| + b_m`, phase `φ_ψ = arg ψ + b_φ` (learned **phase bias**);
- **χ gating**: `χ = f_w(|z|) + b`, a magnitude-only path through the same weights;
  `m_z = 0.5 m_ψ + 0.5 χ`;
- `z' = ReLU(BatchNorm(m_z)) · e^{iφ_ψ}` (nonlinearity on magnitude only, phase passed
  through, cf. modReLU of Arjovsky et al. 2016).

Apart from the phase biases, the layer is equivariant to a global phase rotation.

## What they find

- ARI-BG (object grouping, background excluded): 2Shapes 1.000, 3Shapes 1.000, MNIST&Shape
  0.971; Slot Attention on MNIST&Shape 0.089.
- Ablations on ARI-BG: removing the phase bias `b_φ` → 0.100; removing χ gating → 0.074;
  removing BatchNorm → 0.373; removing the output layer `f_out` → 0.939. Both mechanisms that
  couple magnitude and phase are necessary.
- The phases separate objects even though no loss term touches them.

## Implementation details

- Real weights shared between real and imaginary paths; Adam lr 1e-3, 500-step warmup;
  8 seeds.

## Caveats

- Toy grayscale datasets only, ≤3 objects; superseded by Rotating Features (file 12).
- Shows that phase can carry learned structure (binding), not that complex layers improve
  precision on a physically complex signal.
