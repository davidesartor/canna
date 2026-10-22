# Proteina: Scaling Flow-based Protein Structure Generative Models

Geffner et al., 2025, ICLR 2025 Oral · [arXiv:2503.00710](https://arxiv.org/abs/2503.00710)

**Evidence: A.** Peer-reviewed (International Conference on Learning Representations); 108
citations in 1.6 yr (Semantic Scholar, 2026-10-19).

## What they do

A scaled non-equivariant FM transformer for protein backbones.

## What they find

RoPE *increased* the loss and was dropped. Uniform p(t) gave NaNs. Triangle layers: designability 96.4% vs 93.8%. Registers and QK-norm help.

## Implementation details

Registers; a non-uniform t density. Self-conditioning, disabled for fold-conditioned sampling.

## Caveats

Unconditional or fold-conditional, not NPE.
