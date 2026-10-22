# Complex-Valued Phase-Coherent Transformer

Hioki, 2026, preprint · [arXiv:2605.10123](https://arxiv.org/abs/2605.10123); follow-up
"Complex-valued Phase-Coherent Transformers",
[arXiv:2609.22415](https://arxiv.org/abs/2609.22415)

**Evidence: C.** Not peer-reviewed; 1 citation in 0.4 yr (Semantic Scholar, 2026-10-19).

## What they do

Single-author preprints arguing that the failure of complex transformers is the score/softmax,
not the complex arithmetic. The phase-coherent transformer (PCT) layer:

- complex-linear `W_q, W_k, W_v, W_o` (no bias), RoPE on q and k;
- L2-normalise q and k, score `s_ij = √d · Re⟨q̄_i, k̄_j⟩ ∈ [−√d, √d]` (scaled cosine);
- **no softmax**: a per-pair sigmoid gate `α_ij = σ(s_ij + b)`, b learnable, initialised to
  `−log N`; output `W_o Σ_j α_ij v_j` (real weight × complex value; weights do not sum to 1);
- complex RMSNorm (pre-norm), FFN ×4 with modReLU.

Theorem 1 (proof sketch, partly machine-checked in Lean): the layer commutes with a global
phase rotation of all tokens, since the cosine score is invariant and the value path is
complex-linear. The September follow-up isolates the scaled cosine score as the key
ingredient: with an un-normalised score, the same models stay at chance on ListOps and
needle-in-a-haystack under both gate types; a normalised score at too small a scale also fails.

## What they find

Accuracy, PCT / real softmax transformer / complex softmax transformer (paper Table 1;
mid-scale cells dim 256, depth 6, params matched within ±5%; small-scale real dim 184 vs
complex 128):

| Task | PCT | real softmax | complex softmax |
|---|---|---|---|
| Copy d=2000 | 1.00 | 0.10 | 0.08 |
| NIAH L=2048 | 1.00 | 0.00 | 0.00 |
| ListOps L=1024 | 0.854 | 0.146 | 0.104 |
| FFT-MNIST | 0.45 | 0.32 | 0.39 |
| Phase-memory | 0.99 | 0.98 | 0.93 |
| Multi-pitch | 0.87 | 0.81 | 0.82 |
| RadioML L1 / L2 | 0.35 / 0.35 | 0.30 / 0.22 | 0.24 / 0.27 |

- The strongest **real** baseline ("real_screen", the same normalised-score/gated attention in
  real arithmetic) equals PCT on Copy (d=500 and 2000) and NIAH (1.00), is below it on
  ListOps mid (0.698 vs 0.854), FFT-MNIST (0.43 vs 0.45) and multi-pitch (0.85 vs 0.87), and
  **beats it on RadioML** (0.363 / 0.389 vs 0.345 / 0.352, 3 seeds). A sigmoid gate on the
  raw score (real_sigmoid) stays near real_softmax (Copy d=2000 0.10, ListOps 0.177).
- Real MusicNet: all cells converge to F1 = 0.143 (uninformative).
- Follow-up claims: no degradation up to depth 20; log-linear loss over a 61× parameter range;
  a complex screening + recurrence variant solves Path-X.

## Implementation details

- Gate bias init `−log N` keeps the initial total attention mass O(1). Small-scale cells: 4
  heads, head dim 32 (complex) vs 46 (real). RadioML: dim 64, depth 3. Mostly 3 seeds.

## Caveats

- Unreviewed single-author preprints; long-range tasks dominate, and the softmax baselines'
  near-zero scores on Copy/NIAH suggest weak baselines rather than a complex-number effect.
- The author's own real-arithmetic version of the same attention matches or beats the complex
  one on the physical RF task, so the gains are attributable to the normalised score + gate,
  not to complex values.
- Attention weights are real; phase is carried by values only.
