# Complex Transformer: A Framework for Modeling Complex-Valued Sequence

Yang, Ma, Li, Tsai & Salakhutdinov, 2020, ICASSP 2020 ·
[arXiv:1910.10202](https://arxiv.org/abs/1910.10202)

**Evidence: B.** Peer-reviewed (ICASSP 2020); 66 citations in 6.4 yr (Semantic Scholar,
2026-10-19).

## What they do

The first complex transformer. Input `X = A + iB`; `Q = XW_Q`, `K = XW_K`, `V = XW_V` with
**real** weight matrices (so the projections commute with complex scaling but cannot rotate
phase). The product `Q Kᵀ V` (no conjugate) is expanded into real and imaginary parts and each
of the eight terms computed as an ordinary real multi-head attention:

- Re: `MH(A,A,A) − MH(A,B,B) − MH(B,A,B) − MH(B,B,A)`
- Im: `MH(A,A,B) + MH(A,B,A) + MH(B,A,A) − MH(B,B,B)`

where `MH(q,k,v)` uses `Min-Max-Norm(q kᵀ/√d_k) v` (Eq. 5–6): softmax is replaced by min-max
normalization because it "prevents gradient explosion better in our model". The effective
attention is thus a complex weight matrix (`QKᵀ` has real and imaginary parts that both
multiply V), but each of the eight real sub-attentions is normalized separately, so it is not
an exact complex product. It is not equivariant to a global phase either: `QKᵀV` with no
conjugate is cubic in X, so `X → e^{iα}X` would give `e^{3iα}` even before normalization (our
reading). The FFN applies separate real
ReLU MLPs to A and B. Pre-residual layer norm; 6 encoder + 6 decoder blocks, 8 heads.

## What they find

MusicNet transcription (APS %, 128 labels, test recordings 2303/2382/1819):

| Model | Params | APS |
|---|---|---|
| cgRNN | 2.36M | 53.0 |
| Deep Real Network | 10.00M | 69.8 |
| Deep Complex Network | 17.14M | 72.9 |
| Concatenated (real) transformer | 9.79M | 71.30 |
| Complex Transformer | 11.61M | 74.22 |

Private IQ WiFi device-fingerprinting set (1,000 classes, first 1,600 IQ samples): complex
transformer 59.94% vs concatenated (Re‖Im) transformer 57.57%, GRU 50.38%, FFN 47.12%.
Conditional generation loss (predict last 40% from first 60%): MusicNet 0.0492 vs 0.0509
(real), IQ 2.2335 vs 2.2580.

## Implementation details

- Adam lr 1e-3, 64 time steps for MusicNet; 6 encoders, 8 heads.
- The decoder uses diagonally masked complex attention; positional encoding not described.
- The 8-term expansion costs 8 real attention evaluations per head.

## Caveats

- Real baseline has ~16% fewer parameters (9.79M vs 11.61M); the 2.9-point APS gain is not
  parameter-matched. Deep Complex/Real Network counts are the authors' estimates and those
  models were trained on 84 piano classes, so that comparison is not like-for-like.
- Eilers & Jiang (2023) note these numbers are on validation data; on a held-out test split
  the Yang attention scored 0.7088 AP vs 0.7109 for a real transformer, i.e. no gain.
- No seeds/error bars. Neither dataset has a physically meaningful global phase; no
  equivariance analysis.
- Real-valued weights mean a layer can never apply a learned phase rotation; any phase shift
  has to come from mixing the A and B terms.
