# DDT: Decoupled Diffusion Transformer

Wang et al., 2025 · [arXiv:2504.05741](https://arxiv.org/abs/2504.05741)

**Evidence: A.** Not peer-reviewed; 115 citations in 1.5 yr (Semantic Scholar, 2026-10-19).

## What they do

A condition encoder takes (x_t, t, class) and produces a self-condition z_t. A lighter velocity decoder takes (x_t, t, z_t). REPA is applied to the encoder output. The encoder sees x_t, so z_t can be reused across adjacent steps only approximately.

## What they find

- FID at 400K, no CFG, encoder-heavy splits win:
  | Size | Split | DDT | REPA |
  |---|---|---|---|
  | B | 8 En / 4 De | 16.32 | 24.4 |
  | L | 20 En / 4 De | 7.98 | 10.0 |
  | XL | 22 En / 6 De | 6.62 | 7.9 |
- Decoder block type: attention+MLP 16.32, conv+MLP 16.96, MLP+MLP 24.13. The decoder needs token mixing.
- Encoder sharing across sampling steps (XL, FID 1.31): 50% sharing is 1.6× faster at the same 1.31. 87% sharing is 3.0× faster at 1.42, or 1.40 with a DP recompute schedule.
- FID 1.31 at 256 epochs vs REPA's 1.42 at 800.

## Implementation details

Make the split encoder-heavy, more so with scale. Use a dynamic-programming schedule to decide when to recompute the encoder.

## Caveats

Reuse is approximate because the encoder sees x_t. Image-latent setting only.
