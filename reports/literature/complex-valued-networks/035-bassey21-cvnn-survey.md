# A survey of complex-valued neural networks

Bassey, Qian & Li, 2021, arXiv preprint · [arXiv:2101.12249](https://arxiv.org/abs/2101.12249)

**Evidence: A.** Unreviewed preprint with 170 citations in ~6 yr (Semantic Scholar,
2026-10-19).

## What they do

A broad survey of CVNNs: motivation, activation functions, learning algorithms
(Wirtinger/CR calculus, complex backprop, complex Hessians and second-order methods,
complex learning rates, Barzilai–Borwein, derivative-free MVN learning), architectures
(MLP, CNN, RNN, unitary RNN, GAN, Hopfield) and applications (radar, PolSAR, communications,
speech, MRI, cryptography).

## What they find

- **Activation taxonomy** (their Table II): split Type A (Re/Im) and Type B
  (amplitude/phase), fully complex (elementary transcendental functions), non-parametric
  (KAF), energy-based, complex ReLU variants (modReLU-type magnitude bias; separate ReLU on
  Re/Im), nonlinear-phase, linear, absolute value, hybrid, Möbius, cardioid. Liouville's
  theorem: a bounded activation cannot be entire, so "it is not possible to have a CVNN that
  uses squashing activation functions and is entire". Fully complex tanh has poles on the
  imaginary axis and needs input scaling. No single activation is judged best.
- **Cited wins over real nets** (as reported by the survey, not re-checked): complex VGG
  slightly better on CIFAR-10 with fewer parameters; unitary RNN beating LSTM on permuted
  MNIST with < ¼ the parameters and 5–10× faster convergence (LSTM better on ordered); radar
  human-activity CV-CNN up to 99.81 % and better at low SNR; complex GAN better on PolSAR with
  few labels.
- **Fair comparison**: cites that "merely doubling the number of real-valued parameters in
  each layer does not give the equivalent effect"; notes extra expressiveness brings
  overfitting risk; no systematic matching protocol in the literature.
- **Open problems**: initialisation (Rayleigh modulus, uniform phase), regularisation (L1/L2
  do not carry over directly; complex nets "not self regularizing"), library/hardware
  support built for real arithmetic, cost (a complex multiply = 4 real multiplies + 2 adds),
  numerical stability needing input scaling, memory for recurrent backprop.

## Implementation details

Survey; no experiments. Precision (fp16/bf16) is not discussed; optimisers only in the
classical (Wirtinger gradient, complex Newton) sense.

## Caveats

- Breadth over depth; the cited comparisons are mostly unmatched and quoted from primary
  papers without critique. Useful as a map of activations and training algorithms, not as
  evidence.
