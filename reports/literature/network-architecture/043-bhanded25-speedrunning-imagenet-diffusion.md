# Speedrunning ImageNet Diffusion

Bhanded, 2025 · [arXiv:2512.12386](https://arxiv.org/abs/2512.12386)

**Evidence: C.** Not peer-reviewed; 3 citations in 0.8 yr (Semantic Scholar, 2026-10-19). Its
claims are the authors' alone: a hypothesis, not a result.

## What they do

A cumulative speedrun stack for SiT-B.

## What they find

- Stack 33.0 → 3.49: REPA, REG, VAE, SPRINT (largest single step, 10.56 → 4.58), RMSNorm, RoPE, QK-norm, value residual (4.02 → 3.64).
- Muon is catastrophic: 4.09 → 48.70. Prodigy is also worse.
- x₀-prediction is "significantly worse".
- Contrastive FM: −0.03. Dispersive loss: negligible.

## Implementation details

**Value residual** (ResFormer): mix the first block's V into every later block's V.

## Caveats

Single author. The Muon result is probably a learning-rate or scaling-rule issue (compare CMuon's 8.73 → 3.31 swing).
