# Co-Domain Symmetry for Complex-Valued Deep Learning

Singhal, Xing & Yu, 2022, CVPR 2022 · [arXiv:2112.01525](https://arxiv.org/abs/2112.01525)

**Evidence: B.** Peer-reviewed (CVPR 2022); 17 citations in 4.9 yr (OpenAlex, 2026-10-19;
Semantic Scholar rate-limited).

## What they do

Treat multiplication of the whole input by a complex scalar `s ∈ ℂ*` (global gain and phase)
as a *co-domain* symmetry and build layers that are equivariant or invariant to it (CDS
networks):

- **Equivariant conv/linear:** any complex-linear layer commutes with `z → s z` once the bias
  is removed (biases dropped everywhere). Gauss's 3-multiplication trick speeds complex conv by
  ~25%.
- **Equivariant nonlinearity `ℰ{𝒩}`:** compute the channel-mean feature `m̂`, then
  `m̂ ⊙ 𝒩(f ⊙ m̂*)` — rotate features into the frame of their own mean phase, apply any
  nonlinearity 𝒩 there (they use a Generalized Tangent ReLU), rotate back. This lets the
  nonlinearity act on *relative* phase while staying equivariant to the absolute one.
- **Equivariant pooling:** keep the max-magnitude element. **Equivariant batch norm:**
  normalise magnitudes, keep phases.
- **Invariant layers:** Division `|z₁|/(|z₂|+ε) · e^{i(∠z₁−∠z₂)}` (invariant to all of ℂ*);
  Conjugate `z₁ z₂*` (invariant to phase only); invariant prototype-distance classifier head.
- Architectures: Type-I (division layer early → invariant thereafter) and Type-E (equivariant
  throughout, invariant distance at the end).

## What they find

Small networks (~25–35k parameters), accuracy %:

| Method | Params | CIFAR10 | CIFAR100 | SVHN |
|---|---|---|---|---|
| DCN (Trabelsi) | 66,858 | 65.17 | 32.52 | 85.26 |
| SurReal | 35,274 | 50.68 | 23.57 | 80.51 |
| Real-valued CNN | 34,282 | 64.43 | 31.93 | 87.47 |
| CDS Type-I | 24,241 | 70.29 | 40.79 | 89.21 |
| CDS Type-E | 25,745 | 68.14 | 41.44 | 89.67 |

MSTAR SAR (natively complex): 5% train split — real 47.4, SurReal 61.1, DCN 49.8, CDS 55.1
(corrected) / 69.5 (original, buggy mask); 100% — 66.9 / 94.9 / 89.1 / 95.1 / 96.1. DCN's
accuracy drops under random complex scaling of the input while CDS stays flat (figures only,
no table). Phase normalisation as a preprocessing fix for DCN works when the phase distribution
is simple but fails for complicated ones (LAB colour encoding).

## Implementation details

- No biases anywhere (they break equivariance). GTReLU has a phase-gradient mask controlling
  which phase range can change; a 2025 erratum reports the original code had this mask flipped.
- The flipped (buggy) mask gave *higher* MSTAR accuracy, especially with little data; the
  authors attribute it to robustness to noisy phase. The corrected mask converges better on
  CIFAR.

## Caveats

- CIFAR/SVHN are real images mapped to complex via a hand-made colour encoding; the CVPR
  claims rest on tiny networks far below SOTA (≈65–70% on CIFAR10).
- The erratum changes the MSTAR conclusion: with the corrected mask CDS is below SurReal at 5%
  data (55.1 vs 61.1).
- Robustness to complex scaling is shown by figure, not quantified.
- The invariant/equivariant machinery assumes one global `s` per input; a signal whose phase
  reference differs between components (e.g. per-source phases in a superposition) is outside
  the symmetry they enforce.
