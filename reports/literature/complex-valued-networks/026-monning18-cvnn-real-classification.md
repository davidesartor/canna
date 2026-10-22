# Evaluation of complex-valued neural networks on real-valued classification tasks

Mönning & Manandhar, 2018, arXiv preprint ·
[arXiv:1811.12351](https://arxiv.org/abs/1811.12351)

**Evidence: B.** Unreviewed preprint with 32 citations in ~8 yr (Semantic Scholar,
2026-10-19).

## What they do

The cleanest *negative* control: complex vs real MLPs at matched real-parameter count on
data with **no phase** (MNIST, CIFAR-10, CIFAR-100, Reuters topic classification). Depths
k ∈ {0, 2, 4, 8} hidden layers, five activations (identity, tanh, ReLU on Re/Im separately,
`|z|²`, `|z|`), 10 runs per configuration, best run reported. Two matching schemes:

1. **Fixed width**: real nets 64 (MNIST, Reuters) or 128 (CIFAR) units; complex nets
   alternate full and half width (64/32, 128/64) so each layer has ≈ the same number of real
   parameters.
2. **Fixed budget**: both get ≈ 500 k real parameters, widths solved in closed form per depth.

## What they find

Best-of-10 test accuracy, real vs complex (fixed width):

| Task | Real | Complex |
|---|---|---|
| MNIST | 98.16 (ReLU, k=4) | 97.68 (ReLU, k=4) |
| Reuters | 81.17 (tanh, k=0) | 79.92 (`|z|`, k=0) |
| CIFAR-10 | 84.45 (ReLU, k=4) | 68.10 (ReLU, k=4) |
| CIFAR-100 | 24.23 (`|z|`, k=0) | 21.23 (ReLU, k=2) |

- At matched parameters the real nets "consistently outperform" the complex ones on real
  data; close on easy tasks, a large gap on CIFAR-10. In the fixed-budget experiment the
  best runs are within 0.1 point (98.78 real vs 98.70 complex on the MNIST-labelled table)
  and a few individual cells favour complex.
- On real inputs the imaginary weights **track the real weights** during training, so the
  complex net behaves like an average of two similar classifiers; the authors propose this
  as a diagnostic (if Im-weights follow Re-weights, complex is not buying anything).
- Complex nets often **failed to train** with `|z|²` or deep tanh (chance accuracy), and were
  "far more sensitive to initialisation", worsening with size — "a significant and unsolved
  problem". Split ReLU was the most stable complex activation; `|z|` beat `|z|²`.
- Recommendation: use complex networks when the input is naturally complex, the noise lives
  in the complex plane, or complex embeddings can be learned.

## Implementation details

Complex weights and biases; output `sigmoid(|z|²)` or `softmax(|z|²)` + cross-entropy.
Trabelsi complex Glorot init. Adam, 100 epochs, learning rate not reported. Notes that
"unboundedness of the activation functions can cause numerical instability".

## Caveats

- Real-valued data only, imaginary input part zero: by construction the complex arithmetic
  has nothing to exploit, so this bounds the *cost* of complex nets, not their benefit on
  phase data.
- Best-of-10 reporting; no error bars. One table's caption/header disagree on the dataset.
- MLPs only (no CNNs on CIFAR), so absolute accuracies are low.
