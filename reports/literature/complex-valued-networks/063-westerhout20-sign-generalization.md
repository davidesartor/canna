# Generalization properties of neural network approximations to frustrated magnet ground states

Westerhout, Astrakhantsev, Tikhonov, Katsnelson, Bagrov, 2020, Nat. Commun. 11, 1593 ·
[arXiv:1907.08186](https://arxiv.org/abs/1907.08186)

**Evidence: A.** Peer-reviewed (Nature Communications); 131 citations in 7.3 yr (Semantic
Scholar, 2026-10-19).

## What they do

This is a supervised study. The targets are exact ground states, from exact diagonalisation,
of frustrated J1–J2 spin models on Kagome, square and triangular lattices: 24 spins, a basis
of about 2.7 × 10⁶ states, and also 36 spins on a square lattice.

Amplitude and sign get **separate networks**:

- **Amplitude**: regression on log|ψ| with squared error.
- **Sign**: a binary classifier (p(+), cross-entropy, threshold at 0.5).

Each is trained on a fraction ε_train of basis states, drawn with probability ∝ |ψ|² to mimic
Monte Carlo coverage, and tested on the rest. The dense 1-layer, dense 2-layer and 2-layer
CNN architectures are scored by overlap with the exact state. To isolate one network, its
prediction is combined with the exact value of the other part.

## What they find

- **Expressivity is not the problem.** Trained on the full basis, overlaps exceed 0.94 for
  all J2/J1.
- **Generalisation is.** With ε_train = 1%, the overlap "drops abruptly when frustration is
  increased". On Kagome it falls sharply near J2/J1 ≈ 0.51. The square lattice shows a
  smooth dip over 0.4–0.6.
- **"Learning the sign structure is considerably more difficult than learning
  amplitudes."** Deep in the frustrated regime the amplitude net still gives "a decent
  result" while the sign net fails.
- **Recovery has a threshold.** At J2/J1 = 0.6, generalisation returns only with a much
  larger ε_train, and the overlap jumps abruptly with training-set size rather than rising
  smoothly. The critical fraction grows slowly with Hilbert-space dimension.
- **Symmetry helps.** CNNs, with built-in translation symmetry, have much smaller dips than
  dense nets. The authors recommend designing for sign generalisation, for example by
  building in symmetries.

## Implementation details

- Adam with lr 1e-4 to 1e-3, batch 64 or 128, and early stopping on a validation fraction
  as the main regulariser.
- Inputs are ±1 spin strings.
- Sign and amplitude are fully decoupled, so "sign" here is a discrete ±1 target, not a
  continuous phase.

## Caveats

- Small systems and supervised fitting. The link to variational training is argued, not
  shown.
- The sign is binary (real Hamiltonians). A continuous-phase target, or one with a phase
  symmetry, is not covered.
- The explanation for the threshold behaviour is heuristic, by the authors' own account.
- Details are from the ar5iv HTML. Exact numeric overlaps for the sign curve are only in
  the figures.
