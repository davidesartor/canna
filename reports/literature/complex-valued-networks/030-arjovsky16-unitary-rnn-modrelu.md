# Unitary evolution recurrent neural networks

Arjovsky, Shah & Bengio, 2016, ICML · [arXiv:1511.06464](https://arxiv.org/abs/1511.06464)

**Evidence: A.** Peer-reviewed (ICML 2016); 942 citations in ~10 yr (Semantic Scholar,
2026-10-19).

## What they do

An RNN whose hidden-to-hidden matrix is **unitary** (norm-preserving, so no vanishing or
exploding gradients through time), parametrised cheaply as
`W = D₃ R₂ F⁻¹ D₂ Π R₁ F D₁`: diagonal phase matrices `D = diag(e^{iw_j})`, complex
Householder reflections `R = I − 2vv*/‖v‖²`, a fixed permutation Π and the unitary DFT F.
O(n) parameters, O(n log n) per step. Introduces **modReLU**,
`modReLU(z) = ReLU(|z| + b) · z/|z|`, with a learned per-unit bias b.

## What they find

- Complex is needed for the parametrisation: "Optimization with this parameterization
  becomes feasible only when considering hidden states in the complex domain" (a real
  orthogonal analogue would need tied conjugate eigenpairs).
- **Copy memory** (T = 100–500, all ≈ 6.5 k params: RNN 80, LSTM 40, uRNN 128 units): uRNN
  solves it; LSTM and tanh-RNN stay at the memoryless baseline.
- **Adding** (uRNN ≈ 9 k params vs LSTM ≈ 60 k): both beat baseline to T = 400; uRNN lower
  error, noisier curve.
- **Pixel MNIST**: ordered LSTM 98.2 % vs uRNN 95.1 %; **permuted** uRNN 91.4 % vs LSTM 88 %
  with < ¼ of the LSTM's parameters.
- Split ReLU on Re/Im "performed poorly"; the authors suspect keeping the hidden-state phase
  matters for storing information over long horizons, hence modReLU.

## Implementation details

- Complex state stored as concatenated (Re, Im); output is a real linear map of that
  concatenation, so standard losses apply.
- Init: diagonal phases U[−π, π], reflection vectors U[−1, 1], Glorot for input/output maps,
  zero biases; "not very sensitive to initialisation".
- RMSProp, lr 1e-3, decay 0.9; LSTM/RNN needed grad clipping at 1, the uRNN did not.
- **modReLU is phase-equivariant** (`modReLU(e^{iα}z) = e^{iα} modReLU(z)`); it only rescales
  the modulus, with a dead zone `|z| < −b`.

## Caveats

- Toy long-memory tasks; the uRNN loses to the LSTM on ordered MNIST. Comparisons match
  parameters on copy only.
- The restricted unitary family limits capacity (later "full-capacity uRNN" work).
- Wolter & Yao 2018 show the stability argument does not strictly hold with modReLU.
- Relevance for us is the activation (modReLU) and the unitary/phase-preserving idea, not the
  RNN.
