# Scaling Rectified Flow Transformers for High-Resolution Image Synthesis

Esser et al., 2024, ICML 2024 · [arXiv:2403.03206](https://arxiv.org/abs/2403.03206)

**Evidence: A.** Peer-reviewed (International Conference on Machine Learning); 4994 citations
in 2.6 yr (Semantic Scholar, 2026-10-19).

## What they do

SD3. Introduces MMDiT, logit-normal t, and QK-norm.

## What they find

- Ranking: MMDiT > CrossDiT > UViT > DiT (figures).
- A third stream gives a small gain.
- **QK RMSNorm with a learnable gain** fixes bf16 divergence caused by logit growth.
- Full fp32 halves throughput.

## Implementation details

QK-norm *with* a learnable gain. Logit-normal time sampling.

## Caveats

Most comparisons are in figures.
