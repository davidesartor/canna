# Nucleus-Image: Sparse MoE for Image Generation

Akiti et al., 2026 · [arXiv:2604.12163](https://arxiv.org/abs/2604.12163)

**Evidence: C.** Not peer-reviewed; 5 citations in 0.5 yr (Semantic Scholar, 2026-10-19). Its
claims are the authors' alone: a hypothesis, not a result.

## What they do

Text acts only as K/V: no queries, no MLP and no t-modulation on text, computed once per prompt. 17B total / 2B active MoE.

## What they find

No speed numbers for the text path.

## Implementation details

The condition stream reduced to K/V projections only.

## Caveats

No ablation of the design choice.
