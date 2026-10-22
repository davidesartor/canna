# RoFormer: Enhanced Transformer with Rotary Position Embedding

Su et al., 2021 (journal 2024), Neurocomputing ·
[arXiv:2104.09864](https://arxiv.org/abs/2104.09864)

**Evidence: A.** Peer-reviewed (Neurocomputing); 1917 citations in 5.5 yr (OpenAlex,
2026-10-19; Semantic Scholar rate-limited).

## What they do

Rotary position embedding (RoPE), derived explicitly as complex multiplication. Pair the
query/key dimensions into complex numbers; at position m multiply by `e^{imθ}`:
`f_q = (W_q x_m) e^{imθ}`, `f_k = (W_k x_n) e^{inθ}`, so the attention score is

`g = Re[ (W_q x_m) (W_k x_n)* e^{i(m−n)θ} ]`,

a function of the relative offset only. Frequencies `θ_i = 10000^{−2i/d}` per complex pair.
The score is the real part of a complex inner product (the same `Re⟨q,k⟩` form as
complex-transformer attention), with a position-dependent phase inserted; **values are not
rotated** (cf. RoVE, file 04). A bound shows the score magnitude decays with relative
distance ("long-term decay"). Compatible with linear attention (RoPE applied after the
non-negative feature map, denominator left unrotated).

## What they find

- WMT14 En–De: BLEU 27.5 vs 27.3 (Transformer-base).
- GLUE dev vs BERT: mixed — better on MRPC (89.5 vs 88.9), STS-B (87.0 vs 85.8), QQP (86.4 vs
  71.2); worse on SST-2 (90.7 vs 93.5), QNLI (88.0 vs 90.5), MNLI (80.2/79.8 vs 84.6/83.4).
- CAIL2019-SCM long-text matching: 68.29% at 512 tokens (BERT 67.77%, WoBERT 68.10%),
  69.79% at 1024.
- MLM pretraining and Performer/Enwik8: faster convergence shown only as loss curves.

## Implementation details

- Rotation applied to q and k after projection, per head, in 2×2 blocks; no parameters.
- Base 10000 sets the slowest wavelength; later work (see architecture report 33, Men et al.
  2024) ties the base to usable context length.

## Caveats

- The headline gains are small and mixed; RoPE's dominance comes from later large-LM adoption,
  not this paper's benchmarks.
- In the complex view, RoPE inserts a *known* phase `e^{i(m−n)θ}` into a real-valued score; it
  gives no mechanism for the network to infer or be equivariant to an *unknown* data phase.
