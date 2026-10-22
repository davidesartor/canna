# Back to Basics: Let Denoising Generative Models Denoise

Li et al., 2025, tech report · [arXiv:2511.13720](https://arxiv.org/abs/2511.13720)

**Evidence: C.** Not peer-reviewed; 4 citations (OpenAlex, arXiv record only (Semantic Scholar
has no entry), 2026-10-19). Its claims are the authors' alone: a hypothesis, not a result.

## What they do

JiT. In pixel space with large patches, predict x directly.

## What they find

- B/16 (768-d patches): x-prediction 8.62, v-prediction 96.5–126.9, ε-prediction 372–395, across all loss spaces.
- B/4 (48-d patches): all nine combinations give 3.46–6.20.
- EDM preconditioning: 35.49 vs 8.62.
- Bottleneck embedding of width 32–512 is fine.
- RoPE + qk-norm: 7.48 → 6.69. In-context class tokens at block 4: → 5.49.

## Implementation details

x-prediction when token dimension ≫ width. In-context tokens.

## Caveats

The effect appears only for high-dimensional tokens.
