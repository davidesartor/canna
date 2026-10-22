# Rotating Features for Object Discovery

Löwe, Lippe, Locatello & Welling, 2023, NeurIPS 2023 ·
[arXiv:2306.00600](https://arxiv.org/abs/2306.00600)

**Evidence: B.** Peer-reviewed (NeurIPS 2023); 37 citations in 3.4 yr (Semantic Scholar,
2026-10-19).

## What they do

Generalise the CAE (file 11) from complex numbers (2-D rotations) to **n-dimensional
"rotating features"**: every feature is an n-vector whose magnitude encodes presence and whose
orientation encodes object identity. Layer structure follows the CAE (weights applied
identically to each of the n components, magnitude-gated nonlinearity, χ-style magnitude
path), with one deliberate change: the biases are **not** shared across the n rotation
dimensions, so the layers are not rotation-equivariant. The authors say "preliminary
experiments have demonstrated that [an equivariant rotation mechanism] results in inferior
performance". Can be stacked on frozen DINO features for real images.

## What they find

- Larger n separates more objects: on 4Shapes the CAE and n = 2 (the complex case) "struggle
  to distinguish all four objects", n = 6 separates them perfectly; n = 10 reaches ARI-BG
  0.959 ± 0.022 on 10Shapes.
- Pascal VOC MBO_i / MBO_c: Rotating Features + DINO 0.407 / 0.460 vs DINOSAUR-MLP
  0.395 / 0.409, DINO-CAE 0.329 / 0.374, DINOSAUR-Transformer 0.440 / 0.512.
- FoodSeg103 MBO 0.484.

## Implementation details

- Object masks read out by clustering output orientations (k-means or agglomerative);
  trained on reconstruction only.

## Caveats

- Object discovery, not signal processing; relevance is that **breaking the rotation
  symmetry improved performance** in a phase-as-identity model, and that a single 2-D
  phase (complex numbers) was too little capacity for many objects.
- Still behind a transformer-based baseline on VOC.
