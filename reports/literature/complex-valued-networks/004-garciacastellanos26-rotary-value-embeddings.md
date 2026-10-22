# RoVE: Rotary Value Embeddings Attention for Relative Position-dependent Value Pathways

García-Castellanos, Weiler & Bekkers, 2026, workshop on Topology, Algebra and Geometry in
Data Science ·
[arXiv:2606.11275](https://arxiv.org/abs/2606.11275)

**Evidence: C.** Workshop paper; 0 citations in 4 months (OpenAlex, 2026-10-19). Its core
operation is GTA's ([network-architecture/73](../network-architecture/073-miyato24-gta.md) (A), ICLR 2024),
which carries the evidence for rotating values.

## What they do

RoPE rotates queries and keys only; RoVE also rotates the **values** by their key position and
un-rotates the attention output by the query position:

- weights unchanged: `A_ij = softmax_j( (W_Q x_i)ᵀ R_{j−i} (W_K x_j) / √d )`
- output: `y_i = R_i⁻¹ Σ_j A_ij R_j W_V x_j = Σ_j A_ij R_{j−i} W_V x_j`

so the constant value map becomes an offset-dependent kernel `ψ_δ = R_δ W_V` ("attentive
convolution"). In the complex picture of RoPE (2×2 rotation blocks = multiplication by
`e^{iθδ}`), this is attention with **complex, phase-carrying weights** `A_ij e^{iθ(j−i)}`
whose phase is set by relative position rather than learned from content (our restatement;
the paper uses the rotation-matrix language). No new parameters. The layer stays
shift-equivariant. The paper traces the same operation to GTA (Miyato et al. 2024,
geometry-aware multi-view attention), RayRoPE / cameras-as-relative-encodings (2026), RAVEN
(robotics, 2026) and DeepSeek-V4's shared-KV design with inverse output rotation. The
"VO-RoPE" construction attributed to J. Su's blog is not cited here, and we found no paper
documenting it; RoVE is the paper-form reference for rotating V and un-rotating O.

## What they find

GPT-2-style LMs at ~124M and ~354M parameters on FineWebEdu-10B, 1024-token context, identical
except for the value pathway:

- In-distribution Core ICL: 0.1375 → 0.1416 (124M), 0.1664 → 0.1856 (354M).
- Length extrapolation (354M, no YaRN) perplexity at 4k: 840.10 → 311.38; at 16k:
  1630.72 → 583.84. With YaRN at 4k: 48.61 → 18.40; at 16k: 270.98 → 124.82.
- RULER average NLL (YaRN): 6.62 → 4.33 (354M), 6.75 → 5.35 (124M).

## Implementation details

- Apply the same RoPE rotation to V as to K (per head, same frequencies), then apply the
  inverse rotation at the query position to the attention output before the output
  projection. Parameter count and FLOPs are essentially those of RoPE attention.

## Caveats

- Workshop paper, recent; two model sizes, one dataset, base LMs only.
- The mechanism behind the extrapolation gains is described as an untested hypothesis.
- The phase in the weights is a fixed function of position offset, not of the data; it does
  not by itself make the network equivariant to a global data phase. Whether it helps when
  the RoPE axis is a frequency axis (where a value rotation by `θ·Δbin` resembles a
  time-shift phase ramp) is not studied.
