# GTA: A Geometry-Aware Attention Mechanism for Multi-View Transformers

Miyato, Jaeger, Welling & Geiger, 2024, ICLR 2024 ·
[arXiv:2310.10375](https://arxiv.org/abs/2310.10375)

**Evidence: A.** Peer-reviewed (ICLR); 86 citations in 3 yr (Semantic Scholar, 2026-10-19).

## What they do

Attention whose tokens each carry a group element g_i (a camera pose, a 2-D patch position),
encoded as a relative transformation between query and key-value pairs. Each token's
representation ρ_{g_i} (block-diagonal, e.g. 2×2 rotations as in RoPE) acts on q, k **and v**,
and the output is mapped back into the query's frame:

O = P_g ⊚ Attn(P_gᵀ ⊚ Q, P_g⁻¹ ⊚ K, P_g⁻¹ ⊚ V),  P_g = [ρ_{g_1}, …, ρ_{g_n}]

so token i reads Σ_j A_ij ρ_{g_i} ρ_{g_j}⁻¹ v_j: each value arrives transformed by the
relative group element. With rotation blocks this is RoPE plus a rotated value pathway. They
argue RoPE's q/k-only rotation leaves keys and values in mismatched coordinate frames.

## What they find

- **Value transform ablation** (novel-view synthesis, PSNR): without ρ on V, 36.54
  (CLEVR-TR) and 23.77 (MSN-Hard); full GTA 38.99 and 24.58. The authors call removing it "a
  significant drop".
- **Image diffusion** (DiT on ImageNet): FID 6.26 with 2-D RoPE vs 5.87 with GTA. This does
  not isolate the value transform.

## Implementation details

- No new parameters; the extra cost is one block-diagonal rotation of V and one of the
  output per head.
- The same frequencies rotate q, k and v.

## Caveats

- The headline gains are in multi-view geometry, where token frames genuinely differ. The
  1-D/2-D position case (the DiT row) is a single comparison.
- RoVE (García-Castellanos et al. 2026, workshop, tier C) restates the same operation for 1-D
  language models and lists later rediscoveries (RayRoPE, RAVEN, DeepSeek-V4's inverse output
  rotation).
