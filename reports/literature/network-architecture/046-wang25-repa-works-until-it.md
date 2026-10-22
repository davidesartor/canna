# REPA Works Until It Doesn't: Early-Stopped, Holistic Alignment Supercharges Diffusion Training

Wang et al., 2025, NeurIPS 2025 · [arXiv:2505.16792](https://arxiv.org/abs/2505.16792)

**Evidence: B.** Peer-reviewed (Neural Information Processing Systems); 41 citations in 1.4 yr
(Semantic Scholar, 2026-10-19).

## What they do

HASTE. Switch alignment off at a step τ.

## What they find

- SiT-B 21.3 → 19.6 (τ = 100K). SiT-XL 8.1 → 5.3 (τ = 250K).
- Non-monotone: at XL, 400K, τ = 250K is worse (7.3 vs 5.5).
- Attention-map alignment alone works on MM-DiT: 15.42 → 9.63.

## Implementation details

Decay or stop auxiliary alignment.

## Caveats

τ needs tuning.
