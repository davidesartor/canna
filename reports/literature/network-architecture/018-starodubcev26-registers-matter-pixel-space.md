# Registers Matter for Pixel-Space Diffusion Transformers

Starodubcev et al., 2026 · [arXiv:2605.16147](https://arxiv.org/abs/2605.16147)

**Evidence: C.** Not peer-reviewed; 4 citations in 0.4 yr (Semantic Scholar, 2026-10-19). Its
claims are the authors' alone: a hypothesis, not a result.

## What they do

Learned register tokens in pixel DiTs.

## What they find

- pDiT-B/16 with registers in layers 4–11: none 11.1, 4 registers 8.37, 16 → 6.80, 32 → 6.45.
- Registers in layers 0–11 give 11.9, *worse* than none.
- In pixel space they help (7.39 → 5.30); in the RAE latent they **hurt** (6.58 → 7.48).
- Untying adaLN/MLP for registers: 3.71 → 3.41 (+18M parameters).

## Implementation details

Insert registers after the first few blocks, not from block 0.

## Caveats

Not universally helpful (the latent-space result is negative).
