# Complex gated recurrent neural networks

Wolter & Yao, 2018, NeurIPS · [arXiv:1806.08267](https://arxiv.org/abs/1806.08267)

**Evidence: B.** Peer-reviewed (NeurIPS 2018); 72 citations in ~8 yr (Semantic Scholar,
2026-10-19).

## What they do

A complex GRU-style cell (cgRNN): unitary complex state transition (kept on the Stiefel
manifold by a multiplicative Cayley-type update) plus reset/update gates. Gates are
**real-valued** functions of complex pre-activations, "modSigmoid"
`σ(α Re z + β Im z)` with α, β ∈ [0, 1], scaling the state's modulus. Systematic comparison of
complex activations in the recurrence, and of bounded vs unbounded ones.

## What they find

- **Activations**: with unitary transitions modReLU `ReLU(|z|+b) z/|z|` is best on memory and
  adding; without unitarity the bounded Hirose activation `tanh(|z|/m²) e^{i arg z}` is better.
  modReLU's advantage "is present only when the state-transition matrices are kept unitary".
  Neither activation guarantees stability — correcting the uRNN stability proof.
- **Synthetic tasks at ≈ 44 k params each** (uRNN 140, GRU 112, real gated variant 90 units):
  memory T = 250 — uRNN converged in all runs, cgRNN in 75 %, GRU unstable, real gated
  variant never; adding T = 250 — cgRNN converged in all runs, uRNN never, and the real GRU
  "marginally outperforms" the cgRNN.
- **Human3.6M motion** (MAE at 80/160/320/400 ms): cgRNN 0.41/0.73/1.12/1.26 vs GRU
  0.42/0.74/1.12/1.27 with ≈ half the parameters (1.8 M vs 3.4 M).
- MusicNet: 53 % accuracy, below the complex CNN of Trabelsi et al. (72.9 AP).
- Gating introduced instability: the gated model converged on 15–16 of 20 runs.

## Implementation details

Split-complex storage (Re, Im as separate real tensors); real inputs given zero imaginary
part; output a real linear map of (Re, Im). Stiefel updates need a matrix inverse per step;
other weights RMSProp (lr 1e-3, synthetic), Glorot init, gate biases init 4 (gates open).
Gradient clipping for explosions. Precision not stated.

## Caveats

- Parameter matching only on the synthetic tasks; there the real GRU ties or wins on adding.
- The motion result is a parameter-efficiency claim (same accuracy, half the parameters), not
  a matched-capacity win.
- The modSigmoid gate mixes Re and Im with fixed-sign real weights: it is **not**
  phase-equivariant, unlike modReLU.
