# Holographic Transformers for Complex-Valued Signal Processing: Integrating Phase Interference into Self-Attention

Huang et al., 2025, preprint · [arXiv:2509.19331](https://arxiv.org/abs/2509.19331)

**Evidence: C.** Not peer-reviewed; 0 citations in 1.1 yr (OpenAlex, 2026-10-19; Semantic
Scholar rate-limited).

## What they do

A complex transformer whose attention output carries the query–key phase, so values are summed
coherently ("interference") rather than with real weights. For complex q_i, k_j:

- `s_ij = ⟨q_i, k_j⟩` (complex inner product), `Δφ_ij = ∠ s_ij`.
- Similarity `sim_ij = Re(s_ij) / (‖q_i‖ ‖k_j‖)` (a cosine, Eq. 1).
- Score `W_ij = sim_ij/√d_k · exp(−α |Δφ_ij|)`, then a real softmax over j → `α_ij` (Eq. 2).
- **Output `H_i = Σ_j α_ij V_j e^{jΔφ_ij}` (Eq. 3)**: each value is rotated by its q–k phase
  offset before summation. Equivalently the effective attention weight is the complex number
  `α_ij · s_ij/|s_ij|` — real softmax magnitude, phase from the q–k product.

Proposition P2 states: under a global rotation `(Q,K,V) → e^{jθ}(Q,K,V)`, `α_ij` is invariant
and `H_i → e^{jθ} H_i` (U(1) equivariance), with a short proof but no empirical check. Other
components: complex embeddings, complex sinusoidal positional encodings, complex FFN, complex
layer norm and dropout, an L1 "phase smoothness" regularizer on successive phase differences,
and a dual head (complex reconstruction loss + task loss).

## What they find

- PolSAR land-cover classification (Finnish dataset): 97.81% accuracy vs 96.44% for the best
  complex baseline (CV-MsAtViT), 92.65% for the Complex Transformer of Yang et al., 89.71%
  ComplexNN, and **64.94% for a real-valued transformer**.
- COST2100 channel prediction (12-step): at 120 km/h MAE 0.52 / RMSE 0.54 vs 2.88 / 4.14 for
  STEMGNN; at 3 km/h STEMGNN has the lower MAE (4.39 vs 5.58).
- Ablation: removing the coherent (phase-rotated) summation costs 17.25 accuracy points in
  classification (text says 16.45); removing the phase-decay term increases MAE at 3 km/h by
  10.18.
- Robustness: phase jitter σ = 0.4 costs 0.57% accuracy; amplitude noise τ = 0.10 costs 0.15%
  (3 seeds).

## Implementation details

- Complexity O(T² d_k) like standard attention, with constant-factor overhead for complex
  arithmetic. α (phase decay) is a hyperparameter.
- Parameter counts, activation function and initialization are not reported.

## Caveats

- Unreviewed preprint; no parameter counts, so the 33-point gap to the real transformer cannot
  be attributed to complex processing vs capacity/tuning (a real transformer at 65% on PolSAR
  where every complex baseline is ≥89% suggests an under-tuned baseline).
- Equivariance is claimed but the complex positional encodings and layer-norm affine
  parameters, if additive, would break it; the paper does not test it.
- `exp(−α|Δφ|)` down-weights keys whose projected phase disagrees with the query's, so it
  favours attending among tokens that are already phase-aligned; whether that helps when the
  relative phase between tokens drifts steadily (a long coherent sinusoid cut into STFT tokens)
  is untested.
- Only two datasets; ablation variances not reported for all rows.
