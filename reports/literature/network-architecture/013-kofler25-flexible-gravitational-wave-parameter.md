# Flexible Gravitational-Wave Parameter Estimation with Transformers

Kofler et al., 2025, PRL 136, 251401 (2026) · [arXiv:2512.02968](https://arxiv.org/abs/2512.02968)

**Evidence: B.** Peer-reviewed (Physical Review Letters); 10 citations in 0.9 yr (Semantic
Scholar, 2026-10-19).

## What they do

Dingo-T1. Frequency-domain tokens with GLU injection of the frequency bounds into the tokenizer, and no RoPE. A transformer encoder over about 208 tokens runs once; a flow head samples.

## What they find

Data-based masking 4.2% vs random masking 2.5% (gain in their metric). 160M parameters, 9.4 days on 8 A100s. Token size 16 suffices. φ_c is still reconstructed analytically.

## Implementation details

Masking that mimics real data gating.

## Caveats

—
