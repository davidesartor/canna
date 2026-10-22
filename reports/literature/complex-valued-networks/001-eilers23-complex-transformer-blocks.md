# Building Blocks for a Complex-Valued Transformer Architecture

Eilers & Jiang, 2023, ICASSP 2023 · [arXiv:2306.09827](https://arxiv.org/abs/2306.09827)

**Evidence: B.** Peer-reviewed (ICASSP 2023); 35 citations in 3.4 yr (Semantic Scholar,
2026-10-19).

## What they do

A transformer where every tensor stays complex (no split into ℝ² channels): complex linear
Q/K/V/output projections (Trabelsi et al. 2018 complex dense layers), a complex layer norm, and
a family of complex scaled dot-product attentions that are compared head to head. With
`⟨Q,K⟩ = Σ_i Q_i conj(K_i)` and σ the ordinary real softmax over keys:

- **ℂAtt (their main proposal, Eq. 8):** `σ(Re⟨Q,K⟩ / √d_k) · V`. Neglecting magnitudes,
  `Re⟨Q,K⟩` is `Σ cos(φ_i − ψ_i)`, so the score measures per-component phase alignment of query
  and key. The weights are real; phase reaches the output only through the complex values V.
- **AAtt (Eq. 9):** `σ(|⟨Q,K⟩| / √d_k) · V` — magnitude only, invariant to the relative phase
  of q and k.
- **APAtt (Eq. 10):** the AAtt weights multiplied by `sgn(⟨Q,K⟩) = ⟨Q,K⟩/|⟨Q,K⟩|`, i.e. the
  attention weight is complex and carries the q–k phase into the value sum.
- **ℐAtt (Eq. 11):** `σ(Re/√d_k) + i·σ(Im/√d_k)`, softmax applied separately to real and
  imaginary parts — a complex-valued weight, but not a phase-equivariant one.
- Each variant also in a "Q(Kᵀ)" form without the conjugate, following Yang et al. 2020.
  Masking is applied after mapping scores to ℝ.

Complex layer norm (Eqs. 12–16): subtract the complex mean, whiten with the inverse square root
of the 2×2 real covariance of (Re, Im), then scale by a learned positive-definite 2×2 matrix
and shift by a learned complex β (5 real degrees of freedom per feature). Positional encoding
is the standard sine/cosine one.

## What they find

MusicNet automatic music transcription (128 classes, 64-step Fourier-transformed windows;
35,111/2,030/3,897 train/val/test) and a sequence-generation task (43 input steps, 21
generated). Average precision on the test set, conjugated ("Dot-Prod") / Q(Kᵀ) forms:

| Attention | Classification | Seq. generation |
|---|---|---|
| ℂAtt (Re) | 0.7164 / 0.7142 | 0.3272 / 0.3283 |
| APAtt (complex weight) | 0.6965 / 0.6926 | 0.2240 / 0.3231 |
| AAtt (magnitude) | 0.7117 / 0.7099 | 0.3172 / 0.3271 |
| ℐAtt (split softmax) | 0.7070 / 0.7059 | 0.3201 / 0.3236 |
| Yang et al. 2020 | – / 0.7088 | – / 0.3072 |
| Real transformer | 0.7109 | 0.0737 |

- Classification: all variants within ~0.02 AP of the real transformer; the Re-score variant
  is best by 0.0055. The phase-carrying APAtt is the *worst* variant on both tasks (with
  conjugation).
- Sequence generation: the real transformer (Re/Im interleaved, doubled input width) overfits
  heavily and reaches 0.0737 vs 0.327 for ℂAtt; the complex models show little or no
  validation decline over 100 epochs.
- Removing the complex CNN front end drops ℂAtt to 0.5240 / 0.1652, i.e. the embedding matters
  far more than the attention variant.

## Implementation details

- 6 layers, 8 heads, embedding dim 320, FFN hidden 2048, dropout 0.1, lr 1e-4, batch 35,
  100 epochs. Encoder embedding: 4-layer complex CNN + FC; decoder embedding: FC.
- Parameter counts (real scalars): classification ℂ-transformer 14M, Yang et al. 12M, real
  transformer 18M; generation 27M / 20M / 33M.
- The FFN activation and the initialization are not specified in the paper. Code is released
  (linked in the paper).

## Caveats

- Not parameter-matched: the real baseline has *more* parameters, so the generation gap is an
  overfitting/regularization effect, not a capacity one.
- Single run per cell, no seeds or error bars; the classification differences (≤0.02 AP) are
  within what seed noise could produce.
- Only one dataset (music audio, a natively complex Fourier input but with no physically
  meaningful global phase); no test of equivariance to a global phase rotation. ℐAtt and the
  Q(Kᵀ) forms are not phase-equivariant by construction; ℂAtt/AAtt/APAtt are, but the full
  network is not, since the learned 2×2 layer-norm scale, the complex shift β and the
  additive positional encoding each break U(1) equivariance (our reading; the paper does not
  discuss it).
- The only phase-carrying-weight variant tested (APAtt) underperformed; this is one data point,
  not a systematic study.
