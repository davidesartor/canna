# Complex-valued neural networks with nonparametric activation functions

Scardapane, Van Vaerenbergh, Hussain & Uncini, 2020 (arXiv 2018), IEEE Trans. Emerging
Topics in Computational Intelligence · [arXiv:1802.08026](https://arxiv.org/abs/1802.08026)

**Evidence: A.** Peer-reviewed (IEEE TETCI); 100 citations in ~8 yr (Semantic Scholar,
2026-10-19).

## What they do

Learnable activations for CVNNs via **kernel activation functions (KAFs)**: each neuron's
nonlinearity is a kernel expansion `g(s) = Σ_n α_n κ(s, d_n)` over a fixed dictionary
`d_n` with learned mixing coefficients `α_n`.

- **Split KAF**: two real KAFs on Re and Im (shared 1D dictionary, separate α), Gaussian
  kernel `exp(−γ(s−d)²)`; 2D real coefficients per neuron.
- **Fully complex KAF**: `g(z) = Σ_{n,m} α_{nm} κ_C(z, d_n + i d_m)` on a D×D complex grid
  (D² complex coefficients per neuron), with the complex Gaussian kernel
  `exp(−γ(z − d*)²)` or an "independent" kernel built from a real kernel on Re and Im.
- Bandwidth `γ = 1/(6Δ²)` from the grid spacing Δ, then optimised once per layer.

## What they find

- **Channel identification** (complex nonlinear channel; circular and strongly
  non-circular input, 1 hidden layer of 10): KAF and C-KAF nets "consistently outperform"
  real 2R-NNs (tanh/ReLU), split-tanh/ReLU, amplitude–phase (AMP), CReLU and modReLU nets
  (MSE shown only in a figure); the gap "cannot be overcome by increasing the size of the
  other networks". The complex-Gaussian kernel was poor; the independent kernel fixed it.
- **Wind prediction** (R²): split KAF 0.444, independent C-KAF 0.442, modReLU 0.438,
  split ReLU 0.438, real 2R-NN ReLU 0.435, CReLU 0.181. Differences within ±0.015 sd except
  CReLU.
- **MNIST via FFT** (100 largest 2D-FFT coefficients, 3×100 hidden): real net on (Re, Im)
  92.39 %, modReLU CVNN 95.92 %, split-KAF CVNN 97.21 %.

## Implementation details

Activation zoo compared: split tanh/ReLU, AMP `z/(c + |z|/r)` (bounded modulus, phase
kept), CReLU, modReLU (bias init 0.1), KAFs. Complex AdaGrad, minibatch 40, L2 on `|w|²`,
Autograd. No parameter counts are given and **no parameter matching** is claimed for MNIST.

## Caveats

- Unmatched comparisons: on MNIST the real and complex nets have the same widths (so the
  complex one has ≈ 2× the real parameters in hidden layers), which confounds the 3.5-point
  gap. KAFs add 40–128 real parameters per neuron.
- Small shallow networks; the fully complex KAF was not run on MNIST.
- Neither split nor kernel KAFs are phase-equivariant in general; AMP and modReLU are.
