# SimpleFold: Folding Proteins is Simpler than You Think

Wang et al., 2025 · [arXiv:2509.18480](https://arxiv.org/abs/2509.18480)

**Evidence: B.** Not peer-reviewed; 31 citations in 1.1 yr (Semantic Scholar, 2026-10-19).

## What they do

Plain t-conditioned transformer FM for folding, with no trunk.

## What they find

- TM 0.611 at 100M parameters, 0.720 at 3B.
- Step count: 50 steps → 0.481, 200 → 0.677.
- Auxiliary LDDT loss on the one-step x̂: LDDT 0.711 → 0.767, TM unchanged.
- A weight of w(t) = 1/t collapses training.
- Time density 0.98·LogitNormal(0.8, 1.7) + 0.02·U(0, 1).

## Implementation details

A structured auxiliary loss on x̂. A data-end-heavy time density.

## Caveats

It pays for having no trunk in sampling steps.
